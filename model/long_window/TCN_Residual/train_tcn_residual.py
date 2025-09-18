import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# local loss class
from losses import FocalLoss

from model import build_tcn_residual, build_tcn_residual_gated
from utils import next_result_dir, save_json, load_npz_windows, get_confusion_elements


class ValMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        # predict on validation set
        probs = self.model.predict(self.X_val, verbose=0)
        if probs.ndim > 1 and probs.shape[-1] > 1:
            probs = probs[:, 1]
        preds = (probs >= 0.5).astype(int)

        # compute metrics
        auc = roc_auc_score(self.y_val, probs)
        f1 = f1_score(self.y_val, preds, zero_division=0)
        recall = recall_score(self.y_val, preds, zero_division=0)

        # confusion elements for validation set (tn, fp, fn, tp)
        tn, fp, fn, tp = confusion_matrix(self.y_val, preds).ravel()

        # attach to logs and history
        logs = logs or {}
        logs['val_auc_custom'] = auc
        logs['val_f1_custom'] = f1
        logs['val_recall_custom'] = recall
        # Align composite score with CNN pipeline: 0.5*recall + 0.3*f1 + 0.2*auc
        score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
        self.history.append({
            'epoch': int(epoch + 1),
            'auc': float(auc),
            'f1': float(f1),
            'recall': float(recall),
            'score': float(score),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
        })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--windows', default='/home/user/projects/train/train_data/slipce/windows_npz.npz')
    p.add_argument('--window_type', choices=['long', 'short', 'auto'], default='auto', help='which windows to use from the NPZ (long/short/auto)')
    p.add_argument('--arch', choices=['tcn', 'gated'], default='tcn', help='model architecture to use')
    p.add_argument('--attn_units', type=int, default=32, help='attention hidden units for gated attention (None=auto)')
    p.add_argument('--gate_type', choices=['sigmoid', 'scalar', 'vector', 'softmax', 'vector_sigmoid', 'vector_softmax', 'vector_sigmoid_softmax'], default='vector_sigmoid', help='gate type for gated attention (sigmoid|scalar|vector|softmax|vector_sigmoid|vector_softmax|vector_sigmoid_softmax)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--filters', type=int, default=64)
    p.add_argument('--kernel', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--result_dir', default=None, help='result directory; default is the repository TCN_Residual/result folder')
    p.add_argument('--use_layernorm', action='store_true')
    # focal loss & curriculum
    p.add_argument('--focal_alpha', type=float, default=0.25, help='alpha balance for focal loss')
    p.add_argument('--focal_gamma_start', type=float, default=0.0, help='starting gamma for focal loss (curriculum start)')
    p.add_argument('--focal_gamma_end', type=float, default=2.0, help='final gamma for focal loss (curriculum end)')
    p.add_argument('--curriculum_epochs', type=int, default=10, help='number of epochs to anneal gamma from start to end')
    p.add_argument('--no_early_stop', action='store_true', help='disable EarlyStopping callback')
    p.add_argument('--run_seed', type=int, default=None, help='optional random seed for numpy/tf initialization')
    return p.parse_args()


def main():
    args = parse_args()
    # Prefer GPU if available: enable memory growth and print device info
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    pass
            tf.config.set_visible_devices(gpus, 'GPU')
            logical = tf.config.list_logical_devices('GPU')
            print(f'GPU detected: {len(gpus)} physical, {len(logical)} logical -> {[d.name for d in logical]}')
        else:
            print('No GPU detected; using CPU.')
    except Exception as _e:
        print('GPU setup warning:', _e)
    # set optional reproducible seed for runs
    if args.run_seed is not None:
        import numpy as _np
        _np.random.seed(int(args.run_seed))
        tf.random.set_seed(int(args.run_seed))
    X, y = load_npz_windows(args.windows)
    print('Loaded', X.shape, y.shape)

    # Ensure binary labels are 0/1
    y = y.flatten().astype(int)

    # train/val/test split
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=42)
    # now: test 20%, val 10% (0.125 of 80% = 10%)

    input_shape = X.shape[1:]

    # focal gamma variable (annealed by CurriculumCallback)
    gamma_var = tf.Variable(float(args.focal_gamma_start), trainable=False, dtype=tf.float32, name='focal_gamma_var')

    class CurriculumCallback(tf.keras.callbacks.Callback):
        """Anneal focal gamma from start to end over curriculum_epochs and optionally log to TensorBoard."""
        def __init__(self, gamma_variable, start, end, epochs, log_dir=None):
            super().__init__()
            self.gamma_variable = gamma_variable
            self.start = float(start)
            self.end = float(end)
            self.epochs = int(epochs)
            self.log_dir = log_dir
            self._writer = None

        def on_train_begin(self, logs=None):
            if self.log_dir is not None:
                try:
                    self._writer = tf.summary.create_file_writer(self.log_dir)
                except Exception:
                    self._writer = None

        def on_epoch_begin(self, epoch, logs=None):
            if self.epochs <= 1:
                new_gamma = self.end
            else:
                frac = min(1.0, float(epoch) / float(max(1, self.epochs - 1)))
                new_gamma = self.start + (self.end - self.start) * frac
            # assign to variable
            try:
                self.gamma_variable.assign(new_gamma)
            except Exception:
                K = tf.keras.backend
                K.set_value(self.gamma_variable, new_gamma)

            # write scalar to tensorboard if writer present
            if self._writer is not None:
                with self._writer.as_default():
                    tf.summary.scalar('focal_gamma', data=float(new_gamma), step=epoch)
                    try:
                        self._writer.flush()
                    except Exception:
                        pass

            print(f'Curriculum: set focal gamma = {new_gamma:.4f} (epoch {epoch + 1})')

    # build selected architecture
    if args.arch == 'gated':
        model = build_tcn_residual_gated(
            input_shape,
            num_classes=1,
            num_filters=args.filters,
            kernel_size=args.kernel,
            dilations=[1, 2, 4, 8],
            dropout_rate=args.dropout,
            use_layernorm=args.use_layernorm,
            use_causal=True,
            residual_scale=0.1,
            trainable_scale=False,
            attn_units=args.attn_units,
            gate_type=args.gate_type,
        )
    else:
        model = build_tcn_residual(
            input_shape,
            num_classes=1,
            num_filters=args.filters,
            kernel_size=args.kernel,
            dilations=[1, 2, 4, 8],
            dropout_rate=args.dropout,
            use_layernorm=args.use_layernorm,
            use_causal=True,
        )
    # compile with FocalLoss instance that reads gamma from gamma_var
    loss_fn = FocalLoss(alpha=float(args.focal_alpha), gamma_variable=gamma_var, from_logits=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=loss_fn, metrics=[tf.keras.metrics.AUC(name='auc')])
    model.summary()

    # default result dir is the `result` folder next to this script when not provided
    if args.result_dir is None:
        args.result_dir = os.path.join(os.path.dirname(__file__), 'result')
    folder = next_result_dir(args.result_dir)
    print('Writing results to', folder)

    # Use TF checkpoint format (no .h5) to avoid h5py dependency for saving weights
    chk_path = os.path.join(folder, 'best.weights')
    checkpoint = ModelCheckpoint(chk_path, monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True)
    # optionally disable EarlyStopping for full runs
    if args.no_early_stop:
        early = None
    else:
        early = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
    val_metric_cb = ValMetricsCallback(X_val, y_val)
    # pass folder as log_dir so CurriculumCallback can emit tf.summary for gamma
    curriculum_cb = CurriculumCallback(gamma_var, start=args.focal_gamma_start, end=args.focal_gamma_end, epochs=args.curriculum_epochs, log_dir=folder)

    # assemble callbacks list (omit None entries)
    cbs = [checkpoint, curriculum_cb, val_metric_cb]
    if early is not None:
        cbs.insert(1, early)
    hist = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch, validation_data=(X_val, y_val), callbacks=cbs, verbose=2)

    # load best model weights (if checkpoint saved)
    if os.path.exists(chk_path):
        try:
            model.load_weights(chk_path)
        except Exception as e:
            print('Warning: failed to load weights from', chk_path, '->', e)

    # evaluate on test
    probs = model.predict(X_test, verbose=0)
    if probs.ndim > 1 and probs.shape[-1] > 1:
        probs = probs[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, probs))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    tp, fp, fn, tn = get_confusion_elements(y_test, preds)

    # find top 4 epochs from val_metric_cb.history
    top_list = sorted(val_metric_cb.history, key=lambda x: x['score'], reverse=True)[:4]

    results = {
        'test': {'auc': auc, 'f1': f1, 'recall': recall, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
        'top_epochs': top_list,
        'params': {'arch': args.arch, 'filters': args.filters, 'kernel': args.kernel, 'dropout': args.dropout, 'attn_units': args.attn_units, 'gate_type': args.gate_type}
    }

    # build a reproducible command string that shows exactly which CLI args were used
    arg_items = []
    for k, v in vars(args).items():
        if isinstance(v, bool):
            if v:
                arg_items.append(f'--{k}')
        else:
            arg_items.append(f'--{k} {v}')
    cmd_str = f'python {os.path.basename(__file__)} ' + ' '.join(arg_items)
    results['cmd'] = cmd_str

    save_json(os.path.join(folder, 'results.json'), results)

    # write report.md
    with open(os.path.join(folder, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('# TCN_Residual Report\n\n')
        # write CLI command used
        f.write('## Command\n')
        f.write('```\n')
        f.write(cmd_str + '\n')
        f.write('```\n\n')

        f.write('## Test metrics\n')
        f.write(f'- AUC: {auc:.4f}\n')
        f.write(f'- F1: {f1:.4f}\n')
        f.write(f'- Recall: {recall:.4f}\n')
        f.write('## Confusion matrix (TP/FP/FN/TN)\n')
        f.write(f'- TP: {tp}\n')
        f.write(f'- FP: {fp}\n')
        f.write(f'- FN: {fn}\n')
        f.write(f'- TN: {tn}\n')
    f.write('\n')
    f.write('## Top 4 epochs by score = 0.5*recall + 0.3*f1 + 0.2*auc\n')
    for e in top_list:
            # include TP/FP/FN/TN for each epoch if available
            tp_e = e.get('TP')
            fp_e = e.get('FP')
            fn_e = e.get('FN')
            tn_e = e.get('TN')
            if tp_e is not None:
                f.write(f"- epoch {e['epoch']}: auc={e['auc']:.4f}, f1={e['f1']:.4f}, recall={e['recall']:.4f}, score={e['score']:.4f}  TP: {tp_e} FP: {fp_e} FN: {fn_e} TN: {tn_e}\n")
            else:
                f.write(f"- epoch {e['epoch']}: auc={e['auc']:.4f}, f1={e['f1']:.4f}, recall={e['recall']:.4f}, score={e['score']:.4f}\n")

    print('Done. Results saved to', folder)


if __name__ == '__main__':
    main()
