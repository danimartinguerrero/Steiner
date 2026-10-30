"""Ajustes lineales de lambda^2 frente a T (Práctica 2, ondas estacionarias).

Modelo: lambda^2 = T / (mu f^2)  ->  recta y = a x + b con y = lambda^2, x = T.
Mínimos cuadrados ordinarios (sin ponderar), como en las Notes on Uncertainty.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

F = 43.0                      # Hz, frecuencia leída en el estroboscopio
U_F = 1.0 / np.sqrt(12)       # resolución de 1 Hz
U_T = 0.01 / np.sqrt(12)      # resolución supuesta del dinamómetro: 0,01 N
# u(L): semidiferencia de las dos lecturas de cada distancia (38,8/37,6 cm y
# 51,9/49,8 cm); el valor central es la primera lectura
U_L2 = (0.388 - 0.376) / 2
U_L1 = (0.519 - 0.498) / 2

casos = {
    "Cuerda simple, $L_2$": dict(L=0.388, uL=U_L2, T=[0.15, 0.04, 0.02, 0.01, 0.006]),
    "Cuerda simple, $L_1$": dict(L=0.519, uL=U_L1, T=[0.25, 0.07, 0.03, 0.008, 0.002]),
    "Cuerda doble, $L_2$":  dict(L=0.388, uL=U_L2, T=[0.29, 0.08, 0.04, 0.02, 0.014, 0.01]),
}

# Densidad medida con la báscula: m = 0,06 g en l = 49 cm
m, l = 0.06e-3, 0.49
mu_bascula = m / l
u_mu_bascula = mu_bascula * np.hypot((0.01e-3 / np.sqrt(12)) / m, 0.001 / np.sqrt(12) / l)


def ajuste(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    sxx = ((x - x.mean()) ** 2).sum()
    a = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    b = y.mean() - a * x.mean()
    s = np.sqrt(((y - a * x - b) ** 2).sum() / (n - 2))
    ua = s / np.sqrt(sxx)
    ub = ua * np.sqrt((x ** 2).mean())
    r = np.corrcoef(x, y)[0, 1]
    t = stats.t.ppf(0.975, n - 2)
    return dict(a=a, ua=ua, b=b, ub=ub, r=r, t=t, nu=n - 2)


resultados = {}
for nombre, c in casos.items():
    n = np.arange(1, len(c["T"]) + 1)
    lam = 2 * c["L"] / n
    lam2 = lam ** 2
    u_lam2 = 2 * lam2 * c["uL"] / c["L"]
    fit = ajuste(c["T"], lam2)
    mu = 1 / (fit["a"] * F ** 2)
    u_mu = mu * np.hypot(fit["ua"] / fit["a"], 2 * U_F / F)
    fit.update(mu=mu, u_mu=u_mu, n=n, lam2=lam2, u_lam2=u_lam2)
    resultados[nombre] = fit
    print(f"{nombre}: L = {c['L']} m")
    for ni, Ti, li in zip(n, c["T"], lam2):
        print(f"   n={ni}  T={Ti:<6}  lambda^2={li:.4f} m^2")
    print(f"   a = {fit['a']:.4f} ± {fit['ua']:.4f} m^2/N   (U95 = {fit['t']*fit['ua']:.4f}, t = {fit['t']:.3f})")
    print(f"   b = {fit['b']:.5f} ± {fit['ub']:.5f} m^2   (U95 = {fit['t']*fit['ub']:.5f})")
    print(f"   r = {fit['r']:.5f}")
    print(f"   mu = 1/(a f^2) = {mu:.4e} ± {u_mu:.1e} kg/m")

print(f"mu (báscula) = {mu_bascula:.4e} ± {u_mu_bascula:.1e} kg/m")
print(f"pendiente esperada (cuerda simple) 1/(mu f^2) = {1/(mu_bascula*F**2):.3f} m^2/N")

# ------------------------------------------------------------------ gráficas
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
estilos = [("o", "#1B365D"), ("s", "#3B6EA8"), ("^", "#B8800A")]
archivos = ["ajuste_simple_L2", "ajuste_simple_L1", "ajuste_doble_L2"]


def coma(x, fmt):
    return format(x, fmt).replace(".", "{,}")


def dibuja(ax, nombre, fit, mk, col, etiqueta_datos):
    T = casos[nombre]["T"]
    ax.errorbar(T, fit["lam2"], yerr=fit["u_lam2"], xerr=U_T, fmt=mk, color=col, ms=4,
                capsize=2.5, elinewidth=0.9, label=etiqueta_datos)
    xs = np.linspace(0, max(T) * 1.05, 50)
    signo = "+" if fit["b"] >= 0 else "-"
    ax.plot(xs, fit["a"] * xs + fit["b"], color=col, lw=1.2,
            label=(f"$\\lambda^2 = {coma(fit['a'], '.2f')}\\,T"
                   f" {signo} {coma(abs(fit['b']), '.3f')}$"))


def remata(ax, fig, base):
    comas = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(comas)
    ax.yaxis.set_major_formatter(comas)
    ax.set_xlabel("Tensión $T$ (N)")
    ax.set_ylabel(r"$\lambda^2$ (m$^2$)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print("guardado", base + ".pdf/.png")


for (nombre, fit), (mk, col), arch in zip(resultados.items(), estilos, archivos):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    dibuja(ax, nombre, fit, mk, col, "datos")
    ax.set_title(nombre, fontsize=10)
    remata(ax, fig, arch)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for (nombre, fit), (mk, col) in zip(resultados.items(), estilos):
    dibuja(ax, nombre, fit, mk, col, nombre)
remata(ax, fig, "ajustes_conjunto")
