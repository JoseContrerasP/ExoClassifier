import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers, models
import joblib
import config
from pathlib import Path
from typing import Optional, Dict

class ModelTrainer:
    def __init__(self):
        self.models: Dict[str, any] = {}
    
    @property
    def xgb_model(self):
        return self.models.get('xgboost')
    
    @property
    def rf_model(self):
        return self.models.get('random_forest')
    
    @property
    def nn_model(self):
        return self.models.get('neural_network')

    # ------------------ XGBoost ------------------
    def train_xgboost(self, X_train, y_train, X_val, y_val, params: Optional[dict] = None, verbose: bool=False):
        if params is None:
            params = config.XGBOOST_PARAMS.copy()

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        params['scale_pos_weight'] = class_weights[0] / class_weights[1]
        params.update({
            'objective': 'binary:logistic', 
            'eval_metric': 'auc', 
            'random_state': config.RANDOM_STATE,
            'early_stopping_rounds': 10  # Stop if no improvement for 10 rounds
        })

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=verbose
        )
        self.models['xgboost'] = model
        print(f"✓ XGBoost trained (best iteration: {model.best_iteration})")
        return model

    # ------------------ Random Forest ------------------
    def train_random_forest(self, X_train, y_train, params: Optional[dict] = None):
        if params is None:
            params = config.RF_PARAMS.copy()
        params.update({'random_state': config.RANDOM_STATE, 'n_jobs': -1})

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        print("✓ Random Forest trained")
        return model

    # ------------------ Neural Network ------------------
    def train_neural_network(self, X_train, y_train, X_val, y_val, input_dim: int, params: Optional[dict] = None, verbose: int=2):
        if params is None:
            params = config.NN_PARAMS.copy()

        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        for i, units in enumerate(params['layer_sizes']):
            x = layers.Dense(units, kernel_initializer='he_normal')(x)
            x = layers.PReLU()(x)
            x = layers.BatchNormalization()(x)
            drop_rate = params['dropout_rate'] * (0.67 if i == len(params['layer_sizes']) - 1 else 1)
            x = layers.Dropout(drop_rate)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        early_stop = keras.callbacks.EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            class_weight=class_weight_dict,
            callbacks=[early_stop],
            verbose=verbose
        )

        self.models['neural_network'] = model
        print("✓ Neural Network trained")
        return model

    # ------------------ Save Models ------------------
    def save_models(self, save_dir: Optional[str] = None):
        # If save_dir not provided, use paths from config directly
        if save_dir is None:
            xgb_path = config.XGBOOST_MODEL_PATH
            rf_path = config.RF_MODEL_PATH
            nn_path = config.NN_MODEL_PATH
        else:
            # If save_dir provided, use it with just the filenames
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            xgb_path = save_dir / 'xgboost_model.pkl'
            rf_path = save_dir / 'random_forest_model.pkl'
            nn_path = save_dir / 'neural_network_model.h5'

        joblib.dump(self.models['xgboost'], xgb_path)
        joblib.dump(self.models['random_forest'], rf_path)
        self.models['neural_network'].save(nn_path)
        print(f"✓ All models saved to: {Path(xgb_path).parent}")

    # ------------------ Evaluate Models ------------------
    def evaluate_models(self, X_test_scaled, X_test, y_test):
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} Evaluation ---")
            X_eval = X_test_scaled if model_name == 'neural_network' else X_test

            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_eval)[:, 1]
            else:
                y_pred_proba = model.predict(X_eval, verbose=0).flatten()

            y_pred = (y_pred_proba > 0.5).astype(int)

            print(classification_report(y_test, y_pred, target_names=['False Positive', 'Planet']))
            print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
