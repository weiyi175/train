import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
import tensorflow as tf

from model import build_cnn_bilstm, mlp_attention
from losses import FocalLoss
from utils import next_result_dir, save_json, load_npz_windows

# --- Feature alignment quick check (train/test feature_list parity) ---
# Called early in main(). Non-fatal; prints WARNING if mismatch.
# Env: set SKIP_FEATURE_CHECK=1 to bypass.

def _safe_feature_list(npz_path: str):
    try:
        if not os.path.exists(npz_path):
            return None
        d = np.load(npz_path, allow_pickle=True)
        if 'feature_list' in d:
            fl = d['feature_list']
            # feature_list may be stored as object array -> convert to python list
            if isinstance(fl, np.ndarray):
                try:
                    fl = fl.tolist()
                except Exception:
                    pass
            return list(fl)
    except Exception:
        return None
    return None

def verify_feature_alignment(train_npz: str, test_npz: str):
    if os.environ.get('SKIP_FEATURE_CHECK'):
        print('[feature_check] SKIP_FEATURE_CHECK set -> skipping feature alignment check.')
        return
    if not train_npz or not test_npz:
        print('[feature_check] train/test npz path not provided; skip.')
        return
    train_list = _safe_feature_list(train_npz)
    test_list = _safe_feature_list(test_npz)
    if train_list is None:
        print(f"[feature_check] INFO: Cannot read feature_list from train file: {train_npz}")
        return
    if test_list is None:
        print(f"[feature_check] INFO: Cannot read feature_list from test file: {test_npz}")
        return
    if len(train_list) != len(test_list):
        print(f"[feature_check] WARNING: feature count mismatch train={len(train_list)} test={len(test_list)}")
    # Compare ordered
    if train_list == test_list:
        print(f"[feature_check] OK: train/test feature_list identical (n={len(train_list)})")
        return
    # Order or membership differs -> report diff summary
    set_train = set(train_list)
    set_test = set(test_list)
    missing = [f for f in train_list if f not in set_test]
    extra = [f for f in test_list if f not in set_train]
    # Build mapping (position differences)
    moved = []
    common = set_train.intersection(set_test)
    for f in common:
        i_tr = train_list.index(f)
        i_te = test_list.index(f)
        if i_tr != i_te:
            moved.append((f, i_tr, i_te))
    print('[feature_check] WARNING: feature_list differs:')
    if missing:
        print('  - Missing in test:', missing[:15], ('...(+more)' if len(missing) > 15 else ''))
    if extra:
        print('  - Extra in test:', extra[:15], ('...(+more)' if len(extra) > 15 else ''))
    if moved and not (missing or extra):
        # only ordering issue
        print('  - Reordered features (first 15 shown):', moved[:15])
        print('  - Suggestion: regenerate windows to enforce consistent ordering or reorder at load time.')
    else:
        if moved:
            print('  - Also moved features (first 15 shown):', moved[:15])
    print('  - Suggested action: run check_feature_alignment.py to inspect full diff or regenerate test windows.')

class AccumModel(tf.keras.Model):
    """Keras Model with gradient accumulation.

    accumulate_steps: apply optimizer step every N mini-batches, averaging grads.
    Works with multi-input models (e.g., with attention mask).
    """
    def __init__(self, inputs=None, outputs=None, name=None, accumulate_steps: int = 1):
        super().__init__(inputs=inputs, outputs=outputs, name=name)
        self.accumulate_steps = int(max(1, accumulate_steps))
        self._accum_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='accum_step')
        self._accum_grads = None  # will be list[tf.Variable] lazily created

    def train_step(self, data):
        # Unpack data. Supports list/tuple inputs.
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)

        # Lazy-create accumulators as variables
        if self._accum_grads is None:
            self._accum_grads = [tf.Variable(tf.zeros_like(v), trainable=False) for v in train_vars]
        # Some grads could be None (e.g., disconnected); treat as zeros
        for ag, g, v in zip(self._accum_grads, grads, train_vars):
            if g is None:
                g = tf.zeros_like(v)
            ag.assign_add(g)

        step = self._accum_step.assign_add(1)

        def _apply_grads():
            denom = tf.cast(self.accumulate_steps, train_vars[0].dtype)
            avg_grads = [ag / denom for ag in self._accum_grads]
            self.optimizer.apply_gradients(zip(avg_grads, train_vars))
            # reset buffers in-place
            for ag in self._accum_grads:
                ag.assign(tf.zeros_like(ag))
            return 0

        # apply when reaching accumulation steps
        tf.cond(tf.equal(tf.math.floormod(step, self.accumulate_steps), 0), _apply_grads, lambda: 0)

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


class ValMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, X_val_inputs=None, val_mask=None, window_mask_min_mean: float | None = None):
        super().__init__()
        # If model has multiple inputs (e.g., with mask), pass X_val_inputs as list/tuple.
        self.X_val = X_val
        self.X_val_inputs = X_val_inputs  # optional override for predict
        self.y_val = y_val
        self.val_mask = val_mask
        self.window_mask_min_mean = window_mask_min_mean
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        x_in = self.X_val_inputs if self.X_val_inputs is not None else self.X_val
        probs = self.model.predict(x_in, verbose=0)
        if probs.ndim > 1 and probs.shape[-1] > 1:
            probs = probs[:, 1]
        # Flatten to 1D
        probs = np.asarray(probs).reshape(-1)
        # Optional window-level gating: only allow positive if mean(mask) >= threshold
        if self.window_mask_min_mean is not None and self.val_mask is not None:
            try:
                m_mean = np.asarray(self.val_mask).mean(axis=1)
                gate = (m_mean >= float(self.window_mask_min_mean)).astype(probs.dtype)
                probs = probs * gate
            except Exception:
                pass
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(self.y_val, probs)
        f1 = f1_score(self.y_val, preds, zero_division=0)
        recall = recall_score(self.y_val, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(self.y_val, preds).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        logs = logs or {}
        logs['val_auc_custom'] = auc
        logs['val_f1_custom'] = f1
        logs['val_recall_custom'] = recall
        score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
        logs['val_composite'] = score  # expose composite metric
        precision_aware = 0.5 * precision + 0.3 * f1 + 0.2 * auc
        logs['val_precision_aware'] = precision_aware  # expose precision-aware metric
        self.history.append({
            'epoch': int(epoch + 1),
            'auc': float(auc),
            'f1': float(f1),
            'recall': float(recall),
            'precision': float(precision),
            'score': float(score),
            'precision_aware': float(precision_aware),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
        })


class CurriculumCallback(tf.keras.callbacks.Callback):
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
        try:
            self.gamma_variable.assign(new_gamma)
        except Exception:
            K = tf.keras.backend
            K.set_value(self.gamma_variable, new_gamma)
        if self._writer is not None:
            with self._writer.as_default():
                tf.summary.scalar('focal_gamma', data=float(new_gamma), step=epoch)
                try:
                    self._writer.flush()
                except Exception:
                    pass
        print(f'Curriculum: set focal gamma = {new_gamma:.4f} (epoch {epoch + 1})')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--windows', default='/home/user/projects/train/train_data/slipce/windows_npz.npz')
    p.add_argument('--val_windows', default=None, help='Optional separate validation set NPZ (if provided, disable internal split)')
    p.add_argument('--test_windows', default='/home/user/projects/train/test_data/slipce/windows_npz.npz', help='Optional separate test set NPZ. If provided alone (without --val_windows), internal split provides train/val and this file is used purely as external test. Default set to test_data/slipce/windows_npz.npz')
    p.add_argument('--epochs', type=int, default=70)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--accumulate_steps', type=int, default=8, help='Accumulate gradients over N steps to simulate larger effective batch')
    p.add_argument('--result_dir', default=None)
    p.add_argument('--focal_alpha', type=float, default=0.4)
    p.add_argument('--focal_gamma_start', type=float, default=0.0)
    p.add_argument('--focal_gamma_end', type=float, default=1.0)
    p.add_argument('--curriculum_epochs', type=int, default=20)
    p.add_argument('--no_early_stop', action='store_true')
    p.add_argument('--run_seed', type=int, default=None)
    p.add_argument('--checkpoint_metric', choices=['auc', 'composite', 'precision_aware'], default='auc', help='Metric to monitor for best checkpoint (val_auc or val_composite)')
    # Class weights (optional)
    p.add_argument('--class_weight_neg', type=float, default=1.05, help='Class weight for negative class (label 0)')
    p.add_argument('--class_weight_pos', type=float, default=0.8, help='Class weight for positive class (label 1)')
    # Mask controls
    p.add_argument('--mask_threshold', type=float, default=0.6, help='Attention mask threshold (used by model when use_mask=True)')
    p.add_argument('--mask_mode', choices=['soft', 'hard'], default='soft', help='Mask mode for attention weighting')
    p.add_argument('--window_mask_min_mean', type=float, default=None, help='Window-level gating: only allow positive if mean(mask) >= t')
    return p.parse_args()


def main():
    args = parse_args()
    # Early feature alignment check (if external val/test provided). Use args.windows as train set; val/test precedence for test.
    test_npz_for_check = args.test_windows if args.test_windows else (args.val_windows if args.val_windows else None)
    try:
        if test_npz_for_check:
            verify_feature_alignment(args.windows, test_npz_for_check)
        else:
            print('[feature_check] No external test/val provided -> skipped (internal split will reuse same NPZ).')
    except Exception as e:
        print('[feature_check] ERROR during verification:', e)
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
    if args.run_seed is not None:
        np.random.seed(int(args.run_seed))
        tf.random.set_seed(int(args.run_seed))

    # Show effective batch configuration upfront
    _eff_batch = int(args.batch) * int(max(1, args.accumulate_steps))
    print(f'Config: batch={args.batch}, accumulate_steps={args.accumulate_steps} -> effective_batch={_eff_batch}')

    def derive_mask(path):
        try:
            base = np.load(path, allow_pickle=True)
            feat_names = list(base['feature_list'].tolist()) if 'feature_list' in base else None
            key_raw = 'long_raw' if 'long_raw' in base else ('short_raw' if 'short_raw' in base else None)
            X_raw = base[key_raw] if key_raw else None
            if feat_names is not None and 'occlusion_flag' in feat_names and X_raw is not None:
                ch = feat_names.index('occlusion_flag')
                m = np.asarray(X_raw)[:, ch, :]
                mt = np.clip(m, 0.0, 1.0).astype(np.float32)
                print(f'[mask] {os.path.basename(path)} using occlusion_flag ch={ch}')
                return mt
            elif X_raw is not None:
                mt = np.clip(np.asarray(X_raw)[:, -1, :], 0.0, 1.0).astype(np.float32)
                print(f'[mask] {os.path.basename(path)} using last raw channel (fallback)')
                return mt
        except Exception as e:
            print('[mask] derive failed for', path, '->', e)
        return None

    MASK_MODE = args.mask_mode
    MASK_THRESHOLD = float(args.mask_threshold)
    external_val = args.val_windows is not None
    external_test_only = (args.val_windows is None and args.test_windows is not None)
    if external_val:
        # Load full training set
        X_train, y_train = load_npz_windows(args.windows)
        y_train = y_train.flatten().astype(int)
        print(f'Loaded TRAIN full: {X_train.shape} from {args.windows}')
        # Validation set
        X_val, y_val = load_npz_windows(args.val_windows)
        y_val = y_val.flatten().astype(int)
        print(f'Loaded VAL full: {X_val.shape} from {args.val_windows}')
        # Test (optional)
        if args.test_windows:
            X_test, y_test = load_npz_windows(args.test_windows)
            y_test = y_test.flatten().astype(int)
            print(f'Loaded TEST full: {X_test.shape} from {args.test_windows}')
        else:
            X_test, y_test = X_val, y_val
            print('No --test_windows given; using validation set as test evaluation.')
        input_shape = X_train.shape[1:]

        # --- Time axis alignment (pad/truncate) ---
        def align_time(arr, target_T):
            if arr.shape[1] == target_T:
                return arr
            cur_T = arr.shape[1]
            if cur_T > target_T:
                print(f'[time-align] truncate from T={cur_T} -> {target_T}')
                return arr[:, :target_T, :]
            # pad
            pad_T = target_T - cur_T
            print(f'[time-align] pad from T={cur_T} -> {target_T} (pad {pad_T})')
            pad_block = np.zeros((arr.shape[0], pad_T, arr.shape[2]), dtype=arr.dtype)
            return np.concatenate([arr, pad_block], axis=1)

        train_T = X_train.shape[1]
        if X_val.shape[1] != train_T or X_test.shape[1] != train_T:
            X_val = align_time(X_val, train_T)
            if X_test is X_val:
                # already aligned above
                pass
            else:
                X_test = align_time(X_test, train_T)
            print(f'[time-align] final shapes -> train {X_train.shape}, val {X_val.shape}, test {X_test.shape}')

        m_train = derive_mask(args.windows)
        m_val = derive_mask(args.val_windows)
        m_test = derive_mask(args.test_windows) if args.test_windows else (m_val if X_test is X_val else None)

        # Align masks if needed
        def align_mask(mask, target_T):
            if mask is None:
                return None
            if mask.shape[1] == target_T:
                return mask
            cur_T = mask.shape[1]
            if cur_T > target_T:
                return mask[:, :target_T]
            pad_T = target_T - cur_T
            return np.concatenate([mask, np.zeros((mask.shape[0], pad_T), dtype=mask.dtype)], axis=1)

        if m_train is not None:
            target_T = X_train.shape[1]
            m_train = align_mask(m_train, target_T)
            m_val = align_mask(m_val, target_T)
            m_test = align_mask(m_test, target_T)
        # Consistency: only use mask if all three available with matching lengths
        use_mask = (m_train is not None and m_val is not None and m_test is not None
                    and len(m_train)==len(X_train) and len(m_val)==len(X_val) and len(m_test)==len(X_test))
        if not use_mask:
            m_train = m_val = m_test = None
            print('[mask] Disabled (incomplete across external splits)')
    elif external_test_only:
        # Internal split for train/val from --windows; external test loaded from --test_windows
        X, y = load_npz_windows(args.windows)
        print('Loaded (for internal train/val) ', X.shape, y.shape)
        y = y.flatten().astype(int)
        idx = np.arange(len(X))
        tr_idx, te_idx_internal = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.125, stratify=y[tr_idx], random_state=42)
        X_train, y_train = X[tr_idx2], y[tr_idx2]
        X_val, y_val = X[va_idx], y[va_idx]
        # External test
        X_test, y_test = load_npz_windows(args.test_windows)
        y_test = y_test.flatten().astype(int)
        print(f'Loaded EXTERNAL TEST: {X_test.shape} from {args.test_windows}')
        input_shape = X.shape[1:]
        mask_tensor = derive_mask(args.windows)
        test_mask_tensor = derive_mask(args.test_windows)
        use_mask = (mask_tensor is not None and test_mask_tensor is not None)
        if use_mask:
            m_train, m_val = mask_tensor[tr_idx2], mask_tensor[va_idx]
            m_test = test_mask_tensor
        else:
            m_train = m_val = m_test = None
        # Align time axis of external test to train if mismatch
        def align_time(arr, target_T):
            if arr.shape[1] == target_T:
                return arr
            cur_T = arr.shape[1]
            if cur_T > target_T:
                print(f'[time-align] truncate test from T={cur_T} -> {target_T}')
                return arr[:, :target_T, :]
            pad_T = target_T - cur_T
            print(f'[time-align] pad test from T={cur_T} -> {target_T} (+{pad_T})')
            pad_block = np.zeros((arr.shape[0], pad_T, arr.shape[2]), dtype=arr.dtype)
            return np.concatenate([arr, pad_block], axis=1)
        train_T = X_train.shape[1]
        if X_test.shape[1] != train_T:
            X_test = align_time(X_test, train_T)
            if use_mask and m_test is not None and m_test.shape[1] != train_T:
                # Adjust mask sequence length
                cur_T = m_test.shape[1]
                if cur_T > train_T:
                    m_test = m_test[:, :train_T]
                else:
                    pad_T = train_T - cur_T
                    m_test = np.concatenate([m_test, np.zeros((m_test.shape[0], pad_T), dtype=m_test.dtype)], axis=1)
    else:
        # Original internal stratified split (train/val/test = 0.7/0.1/0.2)
        X, y = load_npz_windows(args.windows)
        print('Loaded', X.shape, y.shape)
        y = y.flatten().astype(int)
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
        tr_idx2, va_idx = train_test_split(tr_idx, test_size=0.125, stratify=y[tr_idx], random_state=42)
        X_train, y_train = X[tr_idx2], y[tr_idx2]
        X_val, y_val = X[va_idx], y[va_idx]
        X_test, y_test = X[te_idx], y[te_idx]
        input_shape = X.shape[1:]
        mask_tensor = derive_mask(args.windows)
        use_mask = mask_tensor is not None
        if use_mask:
            m_train, m_val, m_test = mask_tensor[tr_idx2], mask_tensor[va_idx], mask_tensor[te_idx]
        else:
            m_train = m_val = m_test = None

    base_model = build_cnn_bilstm(
        input_shape,
        num_filters=64,
        kernel_sizes=(3,5,3),
        conv_dropout=0.2,
        lstm_units=64,
        lstm_dropout=0.2,
        attn_units=32,
        use_mask=use_mask,
        mask_mode=MASK_MODE,
        mask_threshold=MASK_THRESHOLD,
    )
    # Wrap with gradient accumulation if needed
    if int(args.accumulate_steps) > 1:
        model = AccumModel(inputs=base_model.inputs, outputs=base_model.outputs, name=base_model.name, accumulate_steps=int(args.accumulate_steps))
    else:
        model = base_model
    print(model.summary())

    gamma_var = tf.Variable(float(args.focal_gamma_start), trainable=False, dtype=tf.float32, name='focal_gamma_var')
    loss_fn = FocalLoss(alpha=float(args.focal_alpha), gamma_variable=gamma_var, from_logits=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=loss_fn, metrics=[tf.keras.metrics.AUC(name='auc')])

    if args.result_dir is None:
        args.result_dir = os.path.join(os.path.dirname(__file__), 'result')
    folder = next_result_dir(args.result_dir)
    print('Writing results to', folder)

    # Use TF checkpoint format (no .h5) to avoid h5py conflicts
    chk_path = os.path.join(folder, 'best.weights')
    if args.checkpoint_metric == 'auc':
        monitor_name = 'val_auc'
    elif args.checkpoint_metric == 'composite':
        monitor_name = 'val_composite'
    else:  # precision_aware
        monitor_name = 'val_precision_aware'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(chk_path, monitor=monitor_name, mode='max', save_best_only=True, save_weights_only=True)
    if args.no_early_stop:
        early = None
    else:
        early = tf.keras.callbacks.EarlyStopping(monitor=monitor_name, mode='max', patience=10, restore_best_weights=True)

    # Prepare validation inputs (with mask if available)
    if use_mask and m_val is not None:
        X_val_inputs = [X_val, m_val]
    else:
        X_val_inputs = X_val

    val_metric_cb = ValMetricsCallback(X_val, y_val, X_val_inputs=X_val_inputs, val_mask=m_val, window_mask_min_mean=args.window_mask_min_mean)
    curriculum_cb = CurriculumCallback(gamma_var, start=args.focal_gamma_start, end=args.focal_gamma_end, epochs=args.curriculum_epochs, log_dir=folder)

    cbs = [checkpoint, curriculum_cb, val_metric_cb]
    if early is not None:
        cbs.insert(1, early)

    # Prepare class weights if provided
    class_weight = None
    if args.class_weight_neg is not None or args.class_weight_pos is not None:
        w_neg = float(args.class_weight_neg) if args.class_weight_neg is not None else 1.0
        w_pos = float(args.class_weight_pos) if args.class_weight_pos is not None else 1.0
        class_weight = {0: w_neg, 1: w_pos}

    if use_mask and m_train is not None:
        hist = model.fit(
            [X_train, m_train], y_train,
            epochs=args.epochs, batch_size=args.batch,
            validation_data=([X_val, m_val], y_val),
            callbacks=cbs, verbose=2,
            class_weight=class_weight,
        )
    else:
        hist = model.fit(
            X_train, y_train,
            epochs=args.epochs, batch_size=args.batch,
            validation_data=(X_val, y_val), callbacks=cbs, verbose=2,
            class_weight=class_weight,
        )

    if os.path.exists(chk_path):
        try:
            model.load_weights(chk_path)
        except Exception as e:
            print('Warning: failed to load weights from', chk_path, '->', e)

    # Save final weights for ensemble averaging
    final_weights_path = os.path.join(folder, 'final.weights')
    try:
        model.save_weights(final_weights_path)
        print(f'Saved final weights to {final_weights_path}')
    except Exception as e:
        print(f'Warning: failed to save final weights -> {e}')

    if use_mask and m_test is not None:
        probs = model.predict([X_test, m_test], verbose=0)
    else:
        probs = model.predict(X_test, verbose=0)
    if probs.ndim > 1 and probs.shape[-1] > 1:
        probs = probs[:, 1]
    # Flatten to 1D
    probs = probs.reshape(-1)
    # Apply window-level gating on test if requested
    if args.window_mask_min_mean is not None and m_test is not None:
        try:
            m_mean = m_test.mean(axis=1)
            gate = (m_mean >= float(args.window_mask_min_mean)).astype(probs.dtype)
            probs = probs * gate
        except Exception:
            pass
    preds = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, probs))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    test_composite = 0.5 * recall + 0.3 * f1 + 0.2 * auc
    test_precision_aware = 0.5 * precision + 0.3 * f1 + 0.2 * auc

    # top4 by custom score
    top_list = sorted(val_metric_cb.history, key=lambda x: x['score'], reverse=True)[:4]
    top_precision_list = sorted(val_metric_cb.history, key=lambda x: x['precision_aware'], reverse=True)[:4]

    eff_batch = int(args.batch) * int(max(1, args.accumulate_steps))
    results = {
        'test': {
            'auc': auc, 'f1': f1, 'recall': recall, 'precision': precision,
            'composite': test_composite, 'precision_aware': test_precision_aware,
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
        },
        'top_epochs': top_list,
        'top_epochs_precision_aware': top_precision_list,
        'params': {
            'model': 'cnn_bilstm_attn', 'batch': int(args.batch), 'accumulate_steps': int(args.accumulate_steps),
            'effective_batch': eff_batch, 'run_seed': (int(args.run_seed) if args.run_seed is not None else None),
            'focal_alpha': float(args.focal_alpha), 'focal_gamma_start': float(args.focal_gamma_start), 'focal_gamma_end': float(args.focal_gamma_end),
            'curriculum_epochs': int(args.curriculum_epochs), 'class_weight_neg': (float(args.class_weight_neg) if args.class_weight_neg is not None else None),
            'class_weight_pos': (float(args.class_weight_pos) if args.class_weight_pos is not None else None), 'mask_mode': MASK_MODE,
            'mask_threshold': MASK_THRESHOLD, 'window_mask_min_mean': (float(args.window_mask_min_mean) if args.window_mask_min_mean is not None else None),
            'checkpoint_metric': args.checkpoint_metric,
        }
    }

    # Compact command summary (mask is auto-derived; no CLI flags for it)
    arg_items = []
    for k, v in vars(args).items():
        if isinstance(v, bool):
            if v:
                arg_items.append(f'--{k}')
        else:
            arg_items.append(f'--{k} {v}')
    cmd_str = f'python {os.path.basename(__file__)} ' + ' '.join(arg_items) + '  [auto-mask: occlusion_flag]'
    results['cmd'] = cmd_str

    save_json(os.path.join(folder, 'results.json'), results)

    with open(os.path.join(folder, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('# CNN_BiLSTM Report\n\n')
        f.write('## Command\n```\n'+cmd_str+'\n```\n\n')
        f.write('## Test metrics\n')
        f.write(f'- AUC: {auc:.4f}\n')
        f.write(f'- F1: {f1:.4f}\n')
        f.write(f'- Recall: {recall:.4f}\n')
        f.write(f'- Precision: {precision:.4f}\n')
        f.write(f'- Composite Score: {test_composite:.4f} (0.5*Recall + 0.3*F1 + 0.2*AUC)\n')
        f.write(f'- Precision-aware Score: {test_precision_aware:.4f} (0.5*Precision + 0.3*F1 + 0.2*AUC)\n')
        f.write('## Confusion matrix (TP/FP/FN/TN)\n')
        f.write(f'- TP: {tp}\n- FP: {fp}\n- FN: {fn}\n- TN: {tn}\n\n')
        f.write('## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)\n')
        for e in top_list:
            f.write(f"- epoch {e['epoch']}: auc={e['auc']:.4f}, f1={e['f1']:.4f}, recall={e['recall']:.4f}, precision={e['precision']:.4f}, score={e['score']:.4f}, precisionAware={e['precision_aware']:.4f}  TP:{e['TP']} FP:{e['FP']} FN:{e['FN']} TN:{e['TN']}\n")
        f.write('\n## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)\n')
        for e in top_precision_list:
            f.write(f"- epoch {e['epoch']}: auc={e['auc']:.4f}, f1={e['f1']:.4f}, recall={e['recall']:.4f}, precision={e['precision']:.4f}, precisionAware={e['precision_aware']:.4f}, composite={e['score']:.4f}  TP:{e['TP']} FP:{e['FP']} FN:{e['FN']} TN:{e['TN']}\n")

    # Note: mask controls
    if use_mask:
        print(f'Mask config -> mode={MASK_MODE}, threshold={MASK_THRESHOLD}, window_gate={args.window_mask_min_mean}')
    print('Done. Results saved to', folder)


if __name__ == '__main__':
    main()
