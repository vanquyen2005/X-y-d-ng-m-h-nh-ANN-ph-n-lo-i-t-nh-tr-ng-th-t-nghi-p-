import numpy as np
import pandas as pd
from typing import Dict

# 🎯 Hàm Sigmoid & đạo hàm
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 🎯 Mô hình mạng nơ-ron nhân tạo (ANN) từ tài liệu Lab5.1 - AI.pdf
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        self.b = []

        # Khởi tạo trọng số & bias theo tài liệu
        for i in range(len(layers) - 1):
            w_ = np.random.randn(layers[i], layers[i + 1]) / layers[i]
            b_ = np.zeros((layers[i + 1], 1))
            self.W.append(w_)
            self.b.append(b_)

    def fit_partial(self, X, y):
        A = [X]

        # Lan truyền xuôi
        for i in range(len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i].T)
            A.append(X)

        # Lan truyền ngược
        y = y.reshape(-1, 1)
        dA = [-(y / A[-1] - (1 - y) / (1 - A[-1]))]
        dW = []
        db = []

        for i in reversed(range(len(self.layers) - 1)):
            dZ = dA[-1] * sigmoid_derivative(A[i + 1])
            dw_ = np.dot(A[i].T, dZ) / X.shape[0]
            db_ = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]
            dA_ = np.dot(dZ, self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)

        dW.reverse()
        db.reverse()

        # Cập nhật trọng số & bias
        for i in range(len(self.layers) - 1):
            self.W[i] -= self.alpha * dW[i]
            self.b[i] -= self.alpha * db[i].reshape(-1, 1)


    def fit(self, X, y, epochs=1000, verbose=100):
        for epoch in range(epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        for i in range(len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i].T)
        return (X >= np.median(X)).astype(int)

    def calculate_loss(self, X, y):
        y_pred = self.predict(X)
        return -np.sum(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))

# 🎯 Xử lý dữ liệu đầu vào
def preprocess_data(file_path: str) -> pd.DataFrame:
    selected_features = [
        "Education", "TotalWorkingYears", "JobRole", "JobLevel", "Department", 
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", 
        "YearsWithCurrManager", "OverTime", "MonthlyIncome", "WorkLifeBalance", 
        "JobSatisfaction", "PerformanceRating", "Attrition"
    ]
    df = pd.read_excel(file_path)
    df = df[selected_features]
    return df

# 🎯 Tính toán thống kê thất nghiệp
def compute_summary(df: pd.DataFrame) -> Dict[str, str]:
    df["Attrition_binary"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
    total = len(df)
    attrition_count = df["Attrition_binary"].sum()
    overall_rate = (attrition_count / total) * 100
    job_attrition = df.groupby("JobRole")["Attrition_binary"].mean()
    highest_job = job_attrition.idxmax()
    lowest_job = job_attrition.idxmin()
    summary = {
        "Total Employees": str(total),
        "Attrition Count": str(attrition_count),
        "Overall Attrition Rate": f"{overall_rate:.2f}%",
        "JobRole with Highest Attrition": highest_job,
        "JobRole with Lowest Attrition": lowest_job
    }
    return summary

# 🎯 Chương trình chính
def main():
    file_path = "Formatted_HR_Employee_Attrition.xlsx"
    df = preprocess_data(file_path)

    X = df.drop(columns=["Attrition"])
    X = pd.get_dummies(X, drop_first=True) 
    y = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0).values.reshape(-1, 1)

    X = (X - X.mean()) / (X.std() + 1e-8)

    print("\n🔹 BẮT ĐẦU HUẤN LUYỆN ANN 🔹")
    model = NeuralNetwork([X.shape[1], 2, 1], alpha=0.1)
    model.fit(X.values, y, epochs=1000, verbose=100)
    predictions = model.predict(X.values)

    accuracy = np.mean(y == predictions)
    print(f"\n📊 Độ chính xác mô hình: {accuracy:.4f}")

    summary = compute_summary(df)

    print("\n📌 Bảng phân loại tình trạng thất nghiệp:")
    print(df[["Education", "TotalWorkingYears", "JobRole", "Attrition"]].head(20))

    print("\n📊 Bảng tỷ lệ thất nghiệp theo trình độ học vấn, kinh nghiệm, ngành nghề:")
    table = df.groupby(["Education", "TotalWorkingYears", "JobRole"])["Attrition"].value_counts(normalize=True).unstack().fillna(0)
    print(table.to_string())  

    print("\n📈 Thống kê tổng quan:")
    print(f"Tổng số nhân viên: {summary['Total Employees']}")
    print(f"Số người thất nghiệp: {summary['Attrition Count']} ({summary['Overall Attrition Rate']})")
    print(f"Ngành có tỷ lệ thất nghiệp cao nhất: {summary['JobRole with Highest Attrition']}")
    print(f"Ngành có tỷ lệ thất nghiệp thấp nhất: {summary['JobRole with Lowest Attrition']}")

if __name__ == "__main__":
    main()
