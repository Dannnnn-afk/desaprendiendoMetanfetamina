import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Cargar/preparar datos
def load_data(path, scale=True):
    df = pd.read_csv(path, usecols=['horsepower','weight','mpg'])
    if not scale:
        return df, None
    # Normalizar con medias/desv std del train; devuelve stats para reutilizar en test
    stats = {
        'hp_mean': df['horsepower'].mean(),
        'hp_std': df['horsepower'].std(ddof=0),
        'w_mean': df['weight'].mean(),
        'w_std': df['weight'].std(ddof=0),
    }
    df_norm = df.copy()
    df_norm['horsepower'] = (df['horsepower'] - stats['hp_mean']) / stats['hp_std']
    df_norm['weight'] = (df['weight'] - stats['w_mean']) / stats['w_std']
    return df_norm, stats

def apply_scale(df, stats):
    if stats is None:
        return df
    df_s = df.copy()
    df_s['horsepower'] = (df['horsepower'] - stats['hp_mean']) / stats['hp_std']
    df_s['weight'] = (df['weight'] - stats['w_mean']) / stats['w_std']
    return df_s

# 2) Predicción
def predict(theta, x1, x2):
    return theta[0] + theta[1]*x1 + theta[2]*x2

# 3) RMSE
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true)**2)))


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true)**2))


def with_bias(x1, x2):
    # Vector de entrada con término de sesgo explícito
    return np.array([1.0, x1, x2], dtype=np.float64)

# 4) Una época de SGD (instancia por instancia, en orden)
def sgd_epoch(theta, df, alpha=1e-6):
    theta = theta.copy()
    for _, row in df.iterrows():
        z = with_bias(row['horsepower'], row['weight'])
        y_hat = float(theta @ z)
        error = y_hat - row['mpg']
        theta -= alpha * error * z
    return theta

# 5) Entrenamiento completo (200 épocas)
def train_sgd(df_train, epochs=200, alpha=1e-6):
    theta = np.zeros(3, dtype=np.float64)  # θ0, θ1, θ2
    history = []
    for _ in range(epochs):
        theta = sgd_epoch(theta, df_train, alpha)
        y_hat = predict(theta, df_train['horsepower'], df_train['weight'])
        history.append(mse(df_train['mpg'], y_hat))
    return theta, history

# 6) Evaluación en prueba: tabla + RMSE
def evaluate(theta, df_test_scaled, df_test_raw=None):
    preds = []
    for _, row in df_test_scaled.iterrows():
        y_hat = predict(theta, row['horsepower'], row['weight'])
        preds.append(y_hat)
    preds = np.array(preds, dtype=np.float64)

    target_df = df_test_raw if df_test_raw is not None else df_test_scaled
    err = preds - target_df['mpg'].to_numpy()
    table = target_df.assign(mpg_est=preds, error=err)
    return table, rmse(target_df['mpg'], preds)


# 7) Grafica pred vs real en prueba
def plot_predictions(table, title='Pred vs Real (test)', show=False):
    plt.figure()
    plt.scatter(table['mpg'], table['mpg_est'], label='predicciones')
    lims = [min(table['mpg'].min(), table['mpg_est'].min()), max(table['mpg'].max(), table['mpg_est'].max())]
    plt.plot(lims, lims, 'k--', label='y = x')
    plt.xlabel('mpg real')
    plt.ylabel('mpg estimado')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    if show:
        plt.show(block=True)


# 7) Pipeline completo: carga, entrenamiento, evaluacion y salida
def run_pipeline(scale=True, alpha=1e-6, epochs=200, show=True):
    train_df, stats = load_data('wrapperAlgoritms/auto_mpg_sgd_train.csv', scale=scale)
    test_df_raw = pd.read_csv('wrapperAlgoritms/auto_mpg_sgd_test.csv', usecols=['horsepower', 'weight', 'mpg'])
    test_df = apply_scale(test_df_raw, stats)

    theta, loss_hist = train_sgd(train_df, epochs=epochs, alpha=alpha)
    table, rmse_test = evaluate(theta, test_df, df_test_raw=test_df_raw)

    # Graficas
    plot_predictions(table, title='Pred vs Real (test)', show=False)
    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel('Epoca')
    plt.ylabel('MSE train')
    plt.title('Historia de perdida (train)')
    plt.grid(True, linestyle=':')
    plt.tight_layout()

    result = {
        'theta': theta,
        'equation': f"y = {theta[0]:.6f} + {theta[1]:.6f}·x1 + {theta[2]:.6f}·x2",
        'test_table': table,
        'rmse_test': rmse_test,
        'loss_history': loss_hist,
    }

    if show:
        plt.show(block=True)

    return result


if __name__ == '__main__':
    result = run_pipeline(scale=True, alpha=1e-6, epochs=5000, show=True)
    print('θ finales:', result['theta'])
    print('Ecuacion:', result['equation'])
    print('Tabla test:')
    print(result['test_table'])
    print('RMSE test:', result['rmse_test'])

# Ejemplo de uso en tu notebook:
# train_df_raw, stats = load_data('wrapperAlgoritms/auto_mpg_sgd_train.csv', scale=True)
# test_df_raw = pd.read_csv('wrapperAlgoritms/auto_mpg_sgd_test.csv', usecols=['horsepower','weight','mpg'])
# train_df = train_df_raw  # ya normalizado si scale=True
# test_df = apply_scale(test_df_raw, stats)  # usa las mismas stats
# theta = train_sgd(train_df, epochs=200, alpha=1e-6)
# tabla, rmse_test = evaluate(theta, test_df)
# print('θ finales:', theta)
# print('Ecuación: y = {:.6f} + {:.6f}·x1 + {:.6f}·x2'.format(*theta))
# print(tabla)
# print('RMSE test:', rmse_test)