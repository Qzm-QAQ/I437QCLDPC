import numpy as np
import math
import matplotlib.pyplot as plt


# ------------------------------
# 1. 构造 QC-LDPC 校验矩阵 H
# ------------------------------

def build_parity_check():
    """
    Build the 51 x 255 parity-check matrix H from the
    3 x 15 exponent matrix E with circulant size Z = 17.
    """
    Z = 17
    m_b, n_b = 3, 15

    # exponent matrix E, exactly as in the report
    E = np.array([
        [0] * 15,
        list(range(15)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11]
    ], dtype=int)

    m, n = m_b * Z, n_b * Z
    H = np.zeros((m, n), dtype=np.uint8)

    # circulant permutation P^shift: position (r, (r+shift) mod Z)
    for br in range(m_b):
        for bc in range(n_b):
            shift = int(E[br, bc] % Z)
            row_base = br * Z
            col_base = bc * Z
            for r in range(Z):
                c = (r + shift) % Z
                H[row_base + r, col_base + c] = 1

    return H


def build_neighbors(H):
    """
    Pre-compute neighbor lists for BP decoding.
    check_neighbors[j] = list of variable indices connected to check j
    var_neighbors[i]   = list of check indices    connected to variable i
    """
    m, n = H.shape
    check_neighbors = [np.where(H[j, :] == 1)[0] for j in range(m)]
    var_neighbors = [np.where(H[:, i] == 1)[0] for i in range(n)]
    return check_neighbors, var_neighbors


# ------------------------------
# 2. log-domain sum-product BP
# ------------------------------

def bp_decode(llr, H, check_neighbors, var_neighbors, max_iter=50):
    """
    Log-domain sum-product decoder.

    llr : channel LLRs, shape (n,)
    H   : parity-check matrix, shape (m, n)
    Return: estimated codeword bits (0/1), shape (n,)
    """
    m, n = H.shape

    # messages indexed as [check, var]
    Q = np.zeros((m, n), dtype=np.float64)  # var -> check
    R = np.zeros((m, n), dtype=np.float64)  # check -> var

    # initialization: Q_{i->j} = L_i for all edges
    for i in range(n):
        neigh = var_neighbors[i]
        if neigh.size > 0:
            Q[neigh, i] = llr[i]

    # iterations
    for _ in range(max_iter):

        # ---- check node update ----
        for j in range(m):
            neigh = check_neighbors[j]
            if neigh.size == 0:
                continue

            Qj = Q[j, neigh]  # messages into check j from all neighbors

            # tanh-based sum-product
            tanh_half = np.tanh(0.5 * Qj)
            prod_all = np.prod(tanh_half)

            for idx, i in enumerate(neigh):
                denom = tanh_half[idx]

                # avoid division by (almost) zero
                if abs(denom) < 1e-12:
                    others = np.delete(tanh_half, idx)
                    val = np.prod(others)
                else:
                    val = prod_all / denom

                # clip to (-1, 1) to avoid atanh overflow
                val = np.clip(val, -0.999999, 0.999999)
                R[j, i] = 2.0 * np.arctanh(val)

        # ---- variable node update ----
        for i in range(n):
            neigh = var_neighbors[i]
            if neigh.size == 0:
                continue

            sumR = np.sum(R[neigh, i])
            for j in neigh:
                Q[j, i] = llr[i] + (sumR - R[j, i])

        # ---- a posteriori decision & syndrome check ----
        T = np.zeros(n, dtype=np.float64)
        for i in range(n):
            neigh = var_neighbors[i]
            if neigh.size > 0:
                T[i] = llr[i] + np.sum(R[neigh, i])
            else:
                T[i] = llr[i]

        c_hat = (T < 0).astype(np.uint8)  # 0 if LLR>=0, 1 if LLR<0

        syndrome = H.dot(c_hat) % 2
        if not syndrome.any():  # all-zero syndrome
            break

    return c_hat


# ------------------------------
# 3. Monte Carlo 仿真
# ------------------------------

def simulate_qc_ldpc(
    EbN0_dB_list,
    R=0.8,
    max_iter=50,
    max_frames=300,
    target_frame_errors=80,
    seed=1234
):
    """
    Simulate BER and FER for the designed QC-LDPC code over BI-AWGN.

    EbN0_dB_list       : list or array of Eb/N0 values in dB
    R                  : code rate (here 0.8)
    max_iter           : max BP iterations
    max_frames         : max frames per SNR point
    target_frame_errors: stop early when this many frame errors collected
    """
    rng = np.random.default_rng(seed)

    H = build_parity_check()
    check_neighbors, var_neighbors = build_neighbors(H)
    m, n = H.shape

    ber_list = []
    fer_list = []

    for snr_db in EbN0_dB_list:
        EbN0_lin = 10 ** (snr_db / 10.0)
        sigma2 = 1.0 / (2 * R * EbN0_lin)
        sigma = math.sqrt(sigma2)

        total_bits = 0
        total_bit_errors = 0
        total_frames = 0
        total_frame_errors = 0

        while total_frames < max_frames and total_frame_errors < target_frame_errors:
            # all-zero codeword -> all +1 BPSK
            x = np.ones(n, dtype=np.float64)

            noise = rng.normal(0.0, sigma, size=n)
            y = x + noise

            # LLR: L_i = 2 y_i / sigma^2
            llr = 2.0 * y / sigma2

            c_hat = bp_decode(llr, H, check_neighbors, var_neighbors, max_iter=max_iter)
            bit_errors = int(c_hat.sum())  # true bits are all 0

            total_bits += n
            total_bit_errors += bit_errors
            total_frames += 1
            if bit_errors > 0:
                total_frame_errors += 1

        ber = total_bit_errors / total_bits
        fer = total_frame_errors / total_frames

        ber_list.append(ber)
        fer_list.append(fer)

        print(
            f"SNR = {snr_db:.2f} dB | "
            f"frames = {total_frames:4d} | "
            f"BER = {ber:.3e} | FER = {fer:.3e}"
        )

    return np.array(ber_list), np.array(fer_list)


# ------------------------------
# 4. 运行仿真并画图
# ------------------------------

if __name__ == "__main__":
    # 和报告里描述类似：0.5 dB ~ 3.5 dB 步长 0.5 dB
    EbN0_dB_list = np.arange(0.5, 3.6, 0.5)

    ber, fer = simulate_qc_ldpc(
        EbN0_dB_list,
        R=0.8,
        max_iter=30,         # 迭代次数，不够可以调大
        max_frames=300,      # 每个点最多多少帧
        target_frame_errors=80
    )

    # 画 BER / FER 曲线（对数坐标）
    plt.figure()
    plt.semilogy(EbN0_dB_list, ber, "o-", label="BER")
    plt.semilogy(EbN0_dB_list, fer, "s-", label="FER")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(r"$E_b/N_0$ [dB]")
    plt.ylabel("Error rate")
    plt.legend()
    plt.tight_layout()

    # 保存为 PDF，和 LaTeX 里的文件名一致
    plt.savefig("ber_fer_qcldpc.pdf")
    plt.show()
