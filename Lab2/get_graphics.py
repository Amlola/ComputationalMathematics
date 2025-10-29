import os
import matplotlib.pyplot as plt

def save_residual_plot(residuals, filename, method_name):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    r_plot = [max(r, 1e-300) for r in residuals]
    iters = list(range(len(r_plot)))

    plt.figure(figsize=(8, 5))
    plt.semilogy(iters, r_plot, linewidth=1.8)
    plt.xlabel("Итерация")
    plt.ylabel("||Ax - b||_2")
    plt.title(f"Убывание невязки: {method_name}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
