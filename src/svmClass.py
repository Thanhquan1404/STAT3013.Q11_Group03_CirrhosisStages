import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# ====================================================================
# CLASS SVM TỔNG QUÁT (KHÔNG CÓ PRINT)
# ====================================================================

class SVMClassifier:
    """
    Class wrapper (bọc) cho sklearn.svm.SVC. (Phiên bản im lặng)
    
    Class này tuân theo giao diện (interface) của sklearn và template
    mẫu (KNNClassifier), chỉ tập trung vào mô hình.
    
    Kernel được truyền vào qua tham số 'kernel' trong __init__.
    
    Tất cả các tham số của SVC (C, kernel, gamma, v.v.)
    đều được hỗ trợ qua __init__, get_params, và set_params.
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 random_state=None, **kwargs):
        """
        Khởi tạo mô hình SVM.
        
        **LƯU Ý:** `probability` được BẬT CỐ ĐỊNH (True) 
        để phương thức .predict_proba() luôn hoạt động.
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.kwargs = kwargs # Cho các tham số khác nếu cần

        # Khởi tạo model gốc của sklearn
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            probability=True, # LUÔN BẬT để .predict_proba() hoạt động
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Cờ (flag) trạng thái
        self.is_fitted = False
        self.classes_ = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """Huấn luyện (fit) mô hình SVM trên dữ liệu."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.classes_ = np.unique(y_train)
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        """Dự đoán nhãn lớp."""
        self._check_fitted()
        return self.model.predict(X)

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """Dự đoán xác suất của các lớp."""
        self._check_fitted()
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def score(self, X, y):
        """Tính độ chính xác (Accuracy) trên X, y."""
        self._check_fitted()
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    # ------------------------------------------------------------------
    def get_params(self, deep=True):
        """Lấy tham số (chuẩn sklearn) để dùng với GridSearchCV."""
        # Trả về các tham số đã lưu
        params = {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'random_state': self.random_state,
        }
        params.update(self.kwargs)
        return params

    # ------------------------------------------------------------------
    def set_params(self, **params):
        """Thiết lập tham số (chuẩn sklearn) để dùng với GridSearchCV."""
        # Cập nhật các thuộc tính của class
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
                
        # Cập nhật lại model bên trong với các tham số mới
        # Lấy tất cả params (bao gồm cả kwargs)
        all_params = self.get_params()
        
        # Lọc ra các tham số mà SVC thực sự chấp nhận
        svc_params = {
            k: v for k, v in all_params.items() 
            if k in SVC()._get_param_names() or k in self.kwargs
        }
        
        # set_params cho model nội bộ
        self.model.set_params(**svc_params)
        return self

    # ------------------------------------------------------------------
    def predict_with_debug(self, X, y_true=None, show_samples=10):
        """
        Trả về dự đoán và xác suất. 
        (Phiên bản này không in ra màn hình).
        """
        self._check_fitted()
        
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        return y_pred, y_proba

    # ------------------------------------------------------------------
    def _check_fitted(self):
        """Kiểm tra xem model đã được fit hay chưa."""
        if not self.is_fitted:
            raise RuntimeError("Model chưa được huấn luyện. Gọi .fit() trước.")