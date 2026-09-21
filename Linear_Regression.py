import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('/Users/knvsaditya/Downloads/RELIANCE_NSE_1994-2025.csv')
df['Next_Close'] = df['Close'].shift(-1)
df = df.dropna()  # last row won't have a next-day close

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Next_Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
print(r2_score(y_test, preds))