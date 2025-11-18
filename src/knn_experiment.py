"""
Experiment với K-Folds validation, ROC curve, confusion matrix và export CSV.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from KNN_module import KNNClassifier


class KNNExperiment:
    """
    KNN experiment với K-Folds Cross-Validation.
    """
    
    def __init__(self, n_neighbors=5, weights='distance', metric='minkowski', p=2, n_folds=5):
        """
        Initialize experiment.
        
        Args:
            n_neighbors: K value
            weights: Vote weighting
            metric: Distance metric
            p: Power parameter
            n_folds: Number of folds for CV
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.n_folds = n_folds
        
        self.fold_results = []
        self.all_y_true = []
        self.all_y_pred = []
        self.all_y_proba = []
        self.classes_ = None
    
    def run(self, X, y):
        """
        Run K-Folds experiment.
        
        Args:
            X: Features (scaled)
            y: Labels
        """
        print("="*80)
        print(f"     KNN K-FOLDS EXPERIMENT | K={self.n_neighbors}, folds={self.n_folds}")
        print("="*80)
        
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n📊 Fold {fold}/{self.n_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            knn = KNNClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                metric=self.metric,
                p=self.p
            )
            knn.fit(X_train, y_train)
            
            # Predictions
            y_pred = knn.predict(X_test)
            y_proba = knn.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            metrics['fold'] = fold
            self.fold_results.append(metrics)
            
            # Store for overall metrics
            self.all_y_true.extend(y_test)
            self.all_y_pred.extend(y_pred)
            self.all_y_proba.extend(y_proba)
            
            # Print fold results
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        self._print_summary()
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all metrics for one fold."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC
        try:
            if len(self.classes_) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
        except:
            metrics['roc_auc'] = np.nan
        
        return metrics
    
    def _print_summary(self):
        """Print overall summary."""
        df = pd.DataFrame(self.fold_results)
        
        print("\n" + "="*80)
        print("                         SUMMARY RESULTS")
        print("="*80)
        
        print(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*80)
        
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if col in df.columns:
                print(f"{col.capitalize():<15} "
                      f"{df[col].mean():<12.4f} "
                      f"{df[col].std():<12.4f} "
                      f"{df[col].min():<12.4f} "
                      f"{df[col].max():<12.4f}")
        
        print("="*80)
    
    def save_metrics_csv(self, filename='knn_metrics.csv'):
        """Save all metrics to CSV."""
        df = pd.DataFrame(self.fold_results)
        
        # Add summary row
        summary = {
            'fold': 'MEAN',
            'accuracy': df['accuracy'].mean(),
            'precision': df['precision'].mean(),
            'recall': df['recall'].mean(),
            'f1_score': df['f1_score'].mean(),
            'roc_auc': df['roc_auc'].mean()
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        # Add std row
        summary_std = {
            'fold': 'STD',
            'accuracy': df['accuracy'][:-1].std(),
            'precision': df['precision'][:-1].std(),
            'recall': df['recall'][:-1].std(),
            'f1_score': df['f1_score'][:-1].std(),
            'roc_auc': df['roc_auc'][:-1].std()
        }
        df = pd.concat([df, pd.DataFrame([summary_std])], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"\n✔️ Metrics saved to: {filename}")
    
    def plot_confusion_matrix(self, filename='confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.all_y_true, self.all_y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes_, yticklabels=self.classes_)
        plt.title(f'Confusion Matrix - KNN (K={self.n_neighbors})', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✔️ Confusion matrix saved to: {filename}")
    
    def plot_roc_curve(self, filename='roc_curve.png'):
        """Plot ROC curve."""
        y_true = np.array(self.all_y_true)
        y_proba = np.array(self.all_y_proba)
        
        plt.figure(figsize=(10, 8))
        
        if len(self.classes_) == 2:
            # Binary classification
            # Chỉ định pos_label cho trường hợp classes = {1, 2}
            pos_label = self.classes_[1]  # Class cao hơn là positive
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=self.classes_)
            
            for i, cls in enumerate(self.classes_):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'Class {cls} (AUC = {roc_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - KNN (K={self.n_neighbors})', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✔️ ROC curve saved to: {filename}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    import warnings
    warnings.filterwarnings('ignore')
    from SMOTE import preprocess_pipeline

  # Cách 1: Sử dụng pipeline hoàn chỉnh (recommended)
    print("\n" + "="*80)
    print("METHOD 1: FULL PIPELINE")
    print("="*80)
    
    df_processed = preprocess_pipeline(
        filepath="W://R-Stats/Liver_Disease_Dataset/ilpd_dataset.csv",
        target_col="Result",
        nan_strategy="median",  # 'median' hoặc 'mean'
        use_smote=False
    )
    
    print(f"\n✔️ Processed data ready!")
    print(f"   Shape: {df_processed.shape}")
    print(f"   Columns: {list(df_processed.columns)}")


    # # ==========================================
    # #   1. LOAD & PREPROCESSING
    # # ==========================================
    # print("="*80)
    # print("KNN TỐI ƯU HÓA CHO DỰ ĐOÁN BỆNH GAN")
    # print("="*80)

    # df = pd.read_csv("W://R-Stats/Liver_Disease_Dataset/d30_dataset.csv", encoding='latin1')
    # df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=False)

    # print(f"\n📊 Dataset Info:")
    # print(f"   - Tổng số mẫu: {len(df)}")
    # print(f"   - Số features: {len(df.columns)-1}")

    # # Encode gender
    # le = LabelEncoder()
    # df['Gender of the patient'] = le.fit_transform(df['Gender of the patient'])

    # # Handle missing với median
    # print(f"\n⚠️  Missing values trước khi xử lý:")
    # missing_before = df.isnull().sum().sum()
    # print(f"   - Tổng: {missing_before} ({missing_before/df.size*100:.2f}%)")

    # for col in df.columns:
    #     if df[col].isnull().sum() > 0:
    #         df[col].fillna(df[col].median(), inplace=True)

    # print(f"   ✓ Đã điền missing values bằng median")
    # print(f"   - Số mẫu giữ được: {len(df)}")

    # Features & Target
    X = df_processed.drop('Result', axis=1).values
    y = df_processed['Result'].values

    print(f"\n📈 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   Class {cls}: {cnt} ({cnt/len(y)*100:.2f}%)")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✂️  Train-Test Split:")
    print(f"   - Train: {len(X_train)} mẫu ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Test:  {len(X_test)} mẫu ({len(X_test)/len(X)*100:.1f}%)")

    # ==========================================
    #   2. TỐI ƯU HÓA HYPERPARAMETERS
    # ==========================================
    print("\n" + "="*80)
    print("2. TÌM HYPERPARAMETERS TỐI ƯU (GridSearchCV)")
    print("="*80)

    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler()
    }

    param_grid = {
        'n_neighbors': range(1, 31, 2),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': [1, 2]
    }

    best_score = 0
    best_scaler_name = None
    best_params = None
    best_scaler = None

    print("\n🔍 Đang thử nghiệm các Scalers...")

    for scaler_name, scaler in scalers.items():
        print(f"\n   Testing {scaler_name}...")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        knn = KNeighborsClassifier()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            knn, param_grid, cv=cv,
            scoring='accuracy', n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_scaled, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_scaler_name = scaler_name
            best_params = grid_search.best_params_
            best_scaler = scaler

        print(f"      → Best CV Score: {grid_search.best_score_:.4f}")
        print(f"      → Best Params: {grid_search.best_params_}")

    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION FOUND:")
    print("="*80)
    print(f"   Scaler:  {best_scaler_name}")
    print(f"   Params:  {best_params}")
    print(f"   CV Score: {best_score:.4f}")

    # ==========================================
    #   3. RUN K-FOLDS EXPERIMENT
    # ==========================================
    print("\n" + "="*80)
    print("3. K-FOLDS EXPERIMENT VỚI BEST CONFIG")
    print("="*80)

    # Scale toàn bộ data với best scaler
    X_scaled = best_scaler.fit_transform(X)

    # Run experiment
    experiment = KNNExperiment(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric'],
        p=best_params['p'],
        n_folds=5
    )
    experiment.run(X_scaled, y)

    # Generate outputs
    experiment.save_metrics_csv('W://ML_Outputs/knn_metrics.csv')
    experiment.plot_confusion_matrix('W://ML_Outputs/confusion_matrix.png')
    experiment.plot_roc_curve('W://ML_Outputs/roc_curve.png')
    print("\nFiles saved to W://ML_Outputs/")

    print("\n" + "="*80)
    print("                    EXPERIMENT COMPLETED")
    print("="*80)