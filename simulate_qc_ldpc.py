import numpy as np
import math
import matplotlib.pyplot as plt


# ============================================================
# 1. 构造 QC-LDPC 校验矩阵 H (51 x 255)
# ============================================================

def build_parity_check():
    """
    构造 QC-LDPC 的校验矩阵 H。
    - 块行数 m_b = 3
    - 块列数 n_b = 15
    - 循环度 Z = 17
    - 指数矩阵 E 决定每个循环矩阵的移位量
    """
    Z = 17
    m_b, n_b = 3, 15  # 块行 / 块列

    # 指数矩阵 E：每个 e_{i,j} 是 [0, Z-1] 里的一个整数
    # 对应一个 ZxZ 的循环置换矩阵的移位量
    E = np.array([
        [0] * 15,
        list(range(15)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 1, 3, 5, 7, 9, 11]
    ], dtype=int)

    m, n = m_b * Z, n_b * Z                       # H 的总行数和总列数
    H = np.zeros((m, n), dtype=np.uint8)          # H 初始化为 0 矩阵

    # 把每个块位置 (br, bc) 展开成一个 ZxZ 循环置换矩阵
    # 循环矩阵 P^shift：第 r 行在列 (r + shift) mod Z 处为 1
    for br in range(m_b):                         # block row 索引
        for bc in range(n_b):                     # block column 索引
            shift = int(E[br, bc] % Z)
            row_base = br * Z                     # 对应的大矩阵起始行
            col_base = bc * Z                     # 对应的大矩阵起始列
            for r in range(Z):
                c = (r + shift) % Z
                H[row_base + r, col_base + c] = 1

    return H


def build_neighbors(H):
    """
    根据 H 预先计算邻接表，方便 BP 使用。

    返回：
    - check_neighbors[j] : 与第 j 个校验节点相连的比特索引列表
    - var_neighbors[i]   : 与第 i 个比特节点相连的校验索引列表
    """
    m, n = H.shape
    check_neighbors = [np.where(H[j, :] == 1)[0] for j in range(m)]
    var_neighbors = [np.where(H[:, i] == 1)[0] for i in range(n)]
    return check_neighbors, var_neighbors


# ============================================================
# 2. log-domain sum-product BP 解码器
# ============================================================

def bp_decode(llr, H, check_neighbors, var_neighbors, max_iter=50):
    """
    对一个码字做 log-domain sum-product (BP) 解码。

    参数：
    - llr : 信道 LLR，形状为 (n,)
    - H   : 校验矩阵，形状为 (m, n)
    - check_neighbors, var_neighbors : 预先算好的邻接表
    - max_iter : 最大迭代次数

    返回：
    - c_hat : 估计出来的码字比特 (0/1)，形状为 (n,)
    """
    m, n = H.shape

    # 消息数组：
    # Q[j, i] = 变量结点 i -> 校验结点 j 的消息
    # R[j, i] = 校验结点 j -> 变量结点 i 的消息
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((m, n), dtype=np.float64)

    # ---------------- 初始化 ----------------
    # 初始时，每条边上的变量到校验消息都设为信道 LLR
    for i in range(n):
        neigh = var_neighbors[i]
        if neigh.size > 0:
            Q[neigh, i] = llr[i]

    # ---------------- 迭代过程 ----------------
    for _ in range(max_iter):

        # ---------- 校验结点更新（check node update） ----------
        for j in range(m):
            neigh = check_neighbors[j]   # 与校验结点 j 相连的比特索引
            if neigh.size == 0:
                continue

            Qj = Q[j, neigh]            # 进入校验结点 j 的所有消息

            # 根据 sum-product 公式，需要计算 tanh(Q/2) 的乘积
            tanh_half = np.tanh(0.5 * Qj)
            prod_all = np.prod(tanh_half)

            for idx, i in enumerate(neigh):
                denom = tanh_half[idx]

                # 为了数值稳定，避免除以 0
                if abs(denom) < 1e-12:
                    # 退化处理：直接把当前这条边剔除后再乘一遍
                    others = np.delete(tanh_half, idx)
                    val = np.prod(others)
                else:
                    val = prod_all / denom

                # arctanh 的输入必须在 (-1, 1) 内，截断一下
                val = np.clip(val, -0.999999, 0.999999)
                R[j, i] = 2.0 * np.arctanh(val)

        # ---------- 变量结点更新（variable node update） ----------
        for i in range(n):
            neigh = var_neighbors[i]
            if neigh.size == 0:
                continue

            # 所有从校验结点回来的消息求和
            sumR = np.sum(R[neigh, i])
            # 对每条边：Q_{i->j} = L_i + sum_{k != j} R_{k->i}
            for j in neigh:
                Q[j, i] = llr[i] + (sumR - R[j, i])

        # ---------- 合并得到 a posteriori LLR，并检查收敛 ----------
        T = np.zeros(n, dtype=np.float64)
        for i in range(n):
            neigh = var_neighbors[i]
            if neigh.size > 0:
                T[i] = llr[i] + np.sum(R[neigh, i])
            else:
                T[i] = llr[i]

        # 硬判决：LLR >= 0 判为 0，否则判为 1
        c_hat = (T < 0).astype(np.uint8)

        # 计算 syndrome 判断是否满足所有校验方程
        syndrome = H.dot(c_hat) % 2
        if not syndrome.any():  # 全 0，说明解码成功，可以提前结束
            break

    return c_hat


# ============================================================
# 3. Monte Carlo 仿真：统计 BER / FER
# ============================================================

def simulate_qc_ldpc(
    EbN0_dB_list,
    R=0.8,
    max_iter=50,
    max_frames=150,
    target_frame_errors=40,
    seed=1234
):
    """
    在 BI-AWGN 信道上，对给定的一组 Eb/N0 点做蒙特卡洛仿真。

    参数：
    - EbN0_dB_list        : Eb/N0 的 dB 列表
    - R                   : 码率
    - max_iter            : BP 最大迭代次数
    - max_frames          : 每个 SNR 点最多仿真的帧数
    - target_frame_errors : 每个 SNR 点累计到这么多错帧就提前停
    - seed                : 随机数种子，方便复现

    返回：
    - ber_list : 每个 SNR 点对应的 BER
    - fer_list : 每个 SNR 点对应的 FER
    """
    rng = np.random.default_rng(seed)

    # 先构造一次 H 和邻接表，在所有 SNR 点共用
    H = build_parity_check()
    check_neighbors, var_neighbors = build_neighbors(H)
    m, n = H.shape

    ber_list = []
    fer_list = []

    for snr_db in EbN0_dB_list:
        # dB -> 线性 Eb/N0
        EbN0_lin = 10 ** (snr_db / 10.0)
        # 由 Eb/N0 与码率 R 计算噪声方差 sigma^2
        sigma2 = 1.0 / (2 * R * EbN0_lin)
        sigma = math.sqrt(sigma2)

        total_bits = 0
        total_bit_errors = 0
        total_frames = 0
        total_frame_errors = 0

        # 在当前 SNR 点循环仿真多帧
        while total_frames < max_frames and total_frame_errors < target_frame_errors:
            # 线性码 + 对称信道：发全零码字即可
            # 全零码字 BPSK 映射后就是全 +1
            x = np.ones(n, dtype=np.float64)

            # AWGN 噪声
            noise = rng.normal(0.0, sigma, size=n)
            y = x + noise

            # 信道 LLR：L_i = 2*y_i / sigma^2
            llr = 2.0 * y / sigma2

            # BP 解码
            c_hat = bp_decode(llr, H, check_neighbors, var_neighbors, max_iter=max_iter)

            # 真值是全 0，所以 1 的个数就是 bit error 数
            bit_errors = int(c_hat.sum())

            total_bits += n
            total_bit_errors += bit_errors
            total_frames += 1
            if bit_errors > 0:
                total_frame_errors += 1

        # 统计 BER / FER
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


# ============================================================
# 4. 主程序：跑仿真并画 BER / FER 曲线
# ============================================================

if __name__ == "__main__":
    # 这里选 0.5 dB ~ 3.5 dB，步长 0.5 dB
    EbN0_dB_list = np.arange(0.5, 3.6, 0.5)

    ber, fer = simulate_qc_ldpc(
        EbN0_dB_list,
        R=0.8,
        max_iter=50,          # 与报告中“最多 50 次迭代”一致
        max_frames=150,       # 每个 SNR 点最多 150 帧
        target_frame_errors=40,
        seed=2025
    )

    # 画 BER / FER 曲线（纵轴对数坐标）
    plt.figure()
    plt.semilogy(EbN0_dB_list, ber, "o-", label="BER")
    plt.semilogy(EbN0_dB_list, fer, "s-", label="FER")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel(r"$E_b/N_0$ [dB]")
    plt.ylabel("Error rate")
    plt.legend()
    plt.tight_layout()

    # 保存!
    plt.savefig("ber_fer_qcldpc.png")
    plt.show()
