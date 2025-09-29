import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

steps = [
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
]

pipe = Pipeline(steps=steps)

param_grid = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [10, 20]
}

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

print("Iniciando o Grid Seach no Pipeline...")
grid_search.fit(X_train, y_train)

print("\nMelhores parâmetros encontrados para o pipeline:")
print(grid_search.best_params_)