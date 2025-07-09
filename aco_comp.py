import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import datetime

# パラメータ
ALPHA = 1.0         # フェロモンの重み
BETA = 1.0         # ヒューリスティック情報の重み
RHO = 0.3           # フェロモン蒸発率
Q = 100             # フェロモンの強さ

# 蟻の数と反復回数
NUM_ANTS = 100
NUM_ITERATIONS = 100

# ---------------------------
# 模擬マップ（Google Maps風）
# ---------------------------
G = nx.Graph()
edges = [
    ('start', 'B', 350), ('start', 'C', 150),
    ('B', 'H', 400), ('C', 'D', 100), ('C', 'E', 250), ('D', 'F', 250),
    ('E', 'F', 100), ('E', 'I', 400), ('F', 'G', 100), ('F', 'J', 400),
    ('G', 'H', 200), ('G', 'K', 400), ('H', 'N', 450), ('I', 'J', 90),
    ('I', 'L', 150), ('J', 'K', 70), ('J', 'goal', 150), ('K', 'M', 90),
    ('L', 'goal', 250), ('M', 'goal', 150), ('M', 'N', 150)
]

# 信号数や交通状況を模擬
traffic_conditions = {edge[:2]: random.uniform(1.0, 2.0) for edge in edges}
#信号の待ち時間と青信号の時間を模擬
signals = {
    ('E', 'I'): {"red_wait": 93.7, "green_duration": 43.5},
    ('F', 'J'): {"red_wait": 93.7, "green_duration": 43.5},
    ('G', 'K'): {"red_wait": 93.7, "green_duration": 43.5},
    ('H', 'N'): {"red_wait": 93.7, "green_duration": 43.5},
    ('I', 'L'): {"red_wait": 30.0, "green_duration": 25.0},
    ('M', 'N'): {"red_wait": 26.8, "green_duration": 32.9},
    # 以下略
}
G.add_weighted_edges_from(edges)

# フェロモンを両方向に持たせる
pheromone = {}
for u, v, _ in edges:
    pheromone[(u, v)] = 1.0
    pheromone[(v, u)] = 1.0

# フェロモンの取得
def get_pheromone(u, v):
    return pheromone.get((u, v)) or pheromone.get((v, u))

# ユーザー好みに応じて重み補正（動的コスト）
def get_dynamic_weight(u, v, mode='distance', hour=8):
    base = G[u][v]['weight']
    traffic = traffic_conditions.get((u, v), 1.0)
    signal_info = signals.get((u, v)) or signals.get((v, u))  # 無向グラフ対応

    # 時間帯による重み調整
    time_factor = 1.5 if hour in range(7, 9) or hour in range(17, 19) else 1.0

    if mode == 'distance':
        return base
    elif mode == 'traffic':
        return base * traffic * time_factor
    elif mode == 'lights':
        if signal_info:
            # 期待される待ち時間（単純に赤信号待ちを加算）
            expected_wait = signal_info['red_wait']**2 / (signal_info['red_wait'] + signal_info['green_duration'])
            weight_penalty = expected_wait * 1.33  # ← ペナルティ調整（ここはチューニングできる）
            return base + weight_penalty
        else:
            return base
    else:
        return base

def path_length(path, mode='distance', hour=8):
    return sum(get_dynamic_weight(path[i], path[i+1], mode, hour) for i in range(len(path) - 1))


def construct_solution(start, end, mode, hour):
    path = [start]
    current = start

    # ループ防止のため最大ステップ数を制限（全ノード数の2倍など）
    max_steps = len(G.nodes) * 2

    steps = 0
    while current != end and steps < max_steps:
        neighbors = list(G.neighbors(current))  # 再訪問を許可
        if not neighbors:
            print(f" {current}から進めるノードがありません。")
            return None

        probabilities = []
        for n in neighbors:
            tau = get_pheromone(current, n) ** ALPHA
            eta = (1.0 / get_dynamic_weight(current, n, mode, hour)) ** BETA
            probabilities.append(tau * eta)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        next_node = random.choices(neighbors, weights=probabilities)[0]

        path.append(next_node)
        current = next_node
        steps += 1

    # 終点に到達していない場合は無効
    if current != end:
        return None

    return path


# ACOで最適化された経路を探索
def aco_optimized_path(start, end, mode , hour=8):
    # フェロモン量
    global pheromone

    # フェロモン濃度を初期化
    pheromone = {edge[:2]: 1.0 for edge in edges}
    #pheromone = {edge[:2]: 1.0 / G[edge[0]][edge[1]]['weight'] for edge in edges}

    # 最適な経路とその長さを初期化
    best_path = None
    best_length = float('inf')

    # 蟻の数と反復回数に基づいて経路を探索
    for _ in range(NUM_ITERATIONS):
        # 蟻が経路を探索
        all_paths = [construct_solution(start, end, mode, hour) for _ in range(NUM_ANTS)]
        all_paths = [p for p in all_paths if p]

        # フェロモンの更新(蒸発)
        for edge in pheromone:
            pheromone[edge] *= (1 - RHO)

        # 各蟻の経路に基づいてフェロモンを更新
        for path in all_paths:
            length = path_length(path, mode, hour)
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                e = edge if edge in pheromone else (edge[1], edge[0])
                pheromone[e] += Q / length

            if length < best_length:
                best_length = length
                best_path = path

    return best_path, best_length

# ---------------------------
# 実行
# ---------------------------
start = 'start'
end = 'goal'
user_mode = 'lights'  # 'distance', 'traffic', 'lights'
departure_hour = 8     # 0〜23時の整数

best_path, best_len = aco_optimized_path(start, end, user_mode, departure_hour)

print(f" 出発: {start} → 目的地: {end}")
print(f" 出発時刻: {departure_hour}:00")
print(f" ユーザーモード: {user_mode}")
print(f" 最短経路: {' → '.join(best_path)}")
print(f" 総コスト（調整後）: {round(best_len, 2)}")

# ---------------------------
# グラフ描画
# ---------------------------


# フェロモン量に基づいてエッジの色を設定
max_pheromone = max(pheromone.values())  # フェロモン量の最大値を取得
min_pheromone = min(pheromone.values())  # フェロモン量の最小値を取得

# フェロモン量を正規化して色の濃さを計算
edge_colors = [
    (pheromone.get((u, v), pheromone.get((v, u), 0)) - min_pheromone) / (max_pheromone - min_pheromone)
    for u, v in G.edges()
]

# グラフ描画
pos = nx.spring_layout(G, seed=44, weight='weight')  # エッジの重みを考慮したレイアウト
nx.draw(
    G, pos, with_labels=True, node_color='lightblue',
    edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=2
)

# エッジラベル（距離）を描画
edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# タイトルを設定
plt.title(f"ACOルート最適化（モード: {user_mode}）")
plt.show()


# フェロモン量をテキストファイルに出力
with open('pheromone_values.txt', mode='w', encoding='utf-8') as file:
    file.write("Edge\t\t\tPheromone\n")
    file.write("-" * 30 + "\n")

    already_written = set()
    for (u, v), value in pheromone.items():
        if (v, u) not in already_written:
            file.write(f"{u} - {v}\t\t{value:.4f}\n")
            already_written.add((u, v))
