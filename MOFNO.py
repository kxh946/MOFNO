import random
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from dataProcessing.ReadDataCSV_new import ReadCSV
from Classfier.KNearestNeighbors import fitnessFunction_KNN_CV


class MOFNO:
    @staticmethod
    def reliefF(X, y, k=5):
        n_samples, n_features = X.shape
        distance = pairwise_distances(X, metric='manhattan')

        weights = np.zeros(n_features)
        penalty = np.zeros(n_features)
        reward = np.zeros(n_features)

        for idx in range(n_samples):
            near_hit = []
            near_miss = dict()
            self_fea = X[idx, :]
            c = np.unique(y).tolist()
            stop_dict = dict()
            for label in c:
                stop_dict[label] = 0
            del c[c.index(y[idx])]

            p_dict = dict()
            p_label_idx = float(len(y[y == y[idx]])) / float(n_samples)
            for label in c:
                p_label_c = float(len(y[y == label])) / float(n_samples)
                p_dict[label] = p_label_c / (1 - p_label_idx)
                near_miss[label] = []

            distance_sort = []
            distance[idx, idx] = np.max(distance[idx, :])
            for i in range(n_samples):
                distance_sort.append([distance[idx, i], int(i), y[i]])
            distance_sort.sort(key=lambda x: x[0])

            for i in range(n_samples):
                if distance_sort[i][2] == y[idx]:
                    if len(near_hit) < k:
                        near_hit.append(distance_sort[i][1])
                    elif len(near_hit) == k:
                        stop_dict[y[idx]] = 1
                else:
                    if len(near_miss[distance_sort[i][2]]) < k:
                        near_miss[distance_sort[i][2]].append(distance_sort[i][1])
                    else:
                        if len(near_miss[distance_sort[i][2]]) == k:
                            stop_dict[distance_sort[i][2]] = 1
                stop = True
                for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
                if stop:
                    break

            near_hit_term = np.zeros(n_features)
            for ele in near_hit:
                near_hit_term = np.array(abs(self_fea - X[ele, :])) + np.array(near_hit_term)

            near_miss_term = dict()
            for (label, miss_list) in near_miss.items():
                near_miss_term[label] = np.zeros(n_features)
                for ele in miss_list:
                    near_miss_term[label] = np.array(abs(self_fea - X[ele, :])) + np.array(near_miss_term[label])
                weights += near_miss_term[label] / (k * p_dict[label])
                reward += near_miss_term[label] / k
            weights -= near_hit_term / k
            penalty += near_hit_term / k

        return weights, penalty, reward

    def pareto_front_2d(self, f1, f2):
        order = np.lexsort((-f2, f1))
        pareto_idx = []
        max_f2 = -np.inf
        for idx in order:
            if f2[idx] > max_f2:
                pareto_idx.append(idx)
                max_f2 = f2[idx]
        return np.array(pareto_idx)

    def get_fused_features(self, penalty, reward, top_percent=0.03):
        n = len(penalty)
        pareto_idx = self.pareto_front_2d(penalty, reward)
        weights = reward - penalty
        k = max(1, int(np.ceil(top_percent * n)))
        top_idx = np.argsort(weights)[-k:]
        fused = np.unique(np.concatenate([pareto_idx, top_idx]))
        is_pareto_fused = np.isin(fused, pareto_idx)
        is_intersection = np.isin(fused, np.intersect1d(pareto_idx, top_idx))
        return fused, is_pareto_fused, is_intersection

    def mofno_init(self, N, d, is_pareto_feature, is_intersection=None, relieff_weights=None):
        pop = []
        if is_intersection is None:
            is_intersection = np.zeros(d, dtype=bool)

        is_intersection = np.asarray(is_intersection).astype(bool)
        is_pareto_feature = np.asarray(is_pareto_feature).astype(bool)

        if relieff_weights is not None:
            relieff_weights = np.asarray(relieff_weights)
            w_min, w_max = relieff_weights.min(), relieff_weights.max()
            if w_max > w_min:
                w_norm = (relieff_weights - w_min) / (w_max - w_min)
            else:
                w_norm = np.full(d, 0.5)
        else:
            w_norm = np.full(d, 0.5)

        score = 0.1 + 0.9 * w_norm
        score[is_intersection] *= 1.5
        score[is_pareto_feature & (~is_intersection)] *= 1.2
        score = score + 0.05
        score = np.maximum(score, 1e-12)
        prob = score / np.sum(score)
        seen = set()
        def to_key(ind):
            return tuple(ind.tolist())
        attempts = 0
        max_attempts = N * 50
        while len(pop) < N and attempts < max_attempts:
            attempts += 1
            ratio = np.random.beta(2, 8)
            ratio = 0.02 + ratio * 0.58
            n_select = int(round(ratio * d))
            n_select = max(1, min(d, n_select))
            ind = np.zeros(d, dtype=int)
            selected_idx = np.random.choice(
                np.arange(d),
                size=n_select,
                replace=False,
                p=prob
            )
            ind[selected_idx] = 1
            key = to_key(ind)
            if key not in seen:
                seen.add(key)
                pop.append(ind)
        while len(pop) < N:
            base = pop[np.random.randint(len(pop))]
            new = base.copy()
            k = np.random.randint(1, max(2, d // 20 + 1))
            flip_idx = np.random.choice(d, size=k, replace=False)
            new[flip_idx] = 1 - new[flip_idx]
            if np.sum(new) == 0:
                new[np.random.choice(d, p=prob)] = 1
            key = to_key(new)
            if key not in seen:
                seen.add(key)
                pop.append(new)
        return np.array(pop[:N])

    def __init__(self, dataX, dataY, dataName, testX=None, testY=None):
        self.dataX = dataX
        self.dataY = dataY
        self.dataName = dataName
        self.testX = testX
        self.testY = testY

        scaler = MinMaxScaler()
        self.dataX = scaler.fit_transform(self.dataX)
        if self.testX is not None:
            self.testX = scaler.transform(self.testX)

        self.weights, self.penalty, self.reward = self.reliefF(self.dataX, self.dataY, k=5)

        fused_idx, self.is_pareto_feature, self.is_intersection = self.get_fused_features(
            self.penalty, self.reward, top_percent=0.03
        )
        self.fused_weights = self.weights[fused_idx]

        self.dataFeature = np.arange(self.dataX.shape[1])[fused_idx]
        self.dataX = self.dataX[:, fused_idx]
        self.dataFeatureNum = len(self.dataFeature)

        if self.testX is not None:
            self.testX = self.testX[:, fused_idx]

        self.nPop = None
        self.MaxIt = None
        self.eta = None
        self.LB = 0.0
        self.UB = 1.0

        self.BestPos = None
        self.BestSolCost = None
        self.BestCosts = None

        self.Position = []
        self.positions = None

        self.generation_best_acc = []
        self.generation_best_nsel = []
        self.pareto_front_history = []
        self.test_pareto_front_history = []
        self.runtime = None
        self.final_train_acc = None
        self.final_test_acc = None
        self.final_n_sel = None

    def setParameter(self, nPop, MaxIt, eta):
        self.nPop = nPop
        self.MaxIt = MaxIt
        self.eta = eta

    def _cost_function(self, position):
        mask = (position >= self.eta).astype(int)
        selected_features = np.where(mask == 1)[0]
        if len(selected_features) == 0:
            return 1.0
        X_selected = self.dataX[:, selected_features]
        acc = fitnessFunction_KNN_CV(findData_x=X_selected, findData_y=self.dataY, CV=5)
        return 1.0 - acc

    def _test_cost_function(self, position):
        mask = (position >= self.eta).astype(int)
        selected_features = np.where(mask == 1)[0]
        n_sel = len(selected_features)
        if n_sel == 0:
            return 1.0, n_sel
        if self.testX is None or self.testY is None:
            return 1.0, n_sel
        knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
        knn.fit(self.dataX[:, selected_features], self.dataY)
        test_acc = knn.score(self.testX[:, selected_features], self.testY)
        return 1.0 - test_acc, n_sel

    def _clipping(self, pos):
        under = pos < self.LB
        over = pos > self.UB
        if np.any(under):
            n_under = np.sum(under)
            pos[under] = self.LB + np.random.uniform(0, 1, n_under) * (self.UB - self.LB)
        if np.any(over):
            n_over = np.sum(over)
            pos[over] = self.LB + np.random.uniform(0, 1, n_over) * (self.UB - self.LB)
        return pos

    def delete_duplicate(self, population):
        if len(population) == 0:
            return population
        masks = np.array([(ind['pos'] >= self.eta).astype(int) for ind in population])
        df = pd.DataFrame(masks)
        df_dup = df.drop_duplicates()
        indices = df_dup.index.values
        return [population[i] for i in indices]

    def _dominates(self, p, q):
        return (p['cost'] <= q['cost'] and p['n_sel'] <= q['n_sel'] and
                (p['cost'] < q['cost'] or p['n_sel'] < q['n_sel']))

    def non_dominated_sort(self, population):
        n = len(population)
        S = [[] for _ in range(n)]
        n_dominated = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif self._dominates(population[j], population[i]):
                    S[j].append(i)
                    n_dominated[i] += 1
            if n_dominated[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dominated[q] -= 1
                    if n_dominated[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        if len(fronts[-1]) == 0:
            fronts.pop()
        return fronts

    def _crowding_distance(self, front, population):
        if len(front) <= 2:
            return {idx: float('inf') for idx in front}

        distances = {idx: 0.0 for idx in front}

        sorted_cost = sorted(front, key=lambda i: population[i]['cost'])
        distances[sorted_cost[0]] = float('inf')
        distances[sorted_cost[-1]] = float('inf')
        f_min = population[sorted_cost[0]]['cost']
        f_max = population[sorted_cost[-1]]['cost']
        if f_max > f_min:
            for k in range(1, len(sorted_cost) - 1):
                i = sorted_cost[k]
                distances[i] += (population[sorted_cost[k + 1]]['cost'] -
                                 population[sorted_cost[k - 1]]['cost']) / (f_max - f_min)

        sorted_sel = sorted(front, key=lambda i: population[i]['n_sel'])
        distances[sorted_sel[0]] = float('inf')
        distances[sorted_sel[-1]] = float('inf')
        f_min = population[sorted_sel[0]]['n_sel']
        f_max = population[sorted_sel[-1]]['n_sel']
        if f_max > f_min:
            for k in range(1, len(sorted_sel) - 1):
                i = sorted_sel[k]
                distances[i] += (population[sorted_sel[k + 1]]['n_sel'] -
                                 population[sorted_sel[k - 1]]['n_sel']) / (f_max - f_min)

        return distances

    def select_next_population(self, combined, nPop):
        if len(combined) <= nPop:
            return combined

        fronts = self.non_dominated_sort(combined)
        selected = []

        for front in fronts:
            if len(selected) + len(front) <= nPop:
                selected.extend(front)
            else:
                distances = self._crowding_distance(front, combined)
                front_sorted = sorted(front, key=lambda i: distances[i], reverse=True)
                remaining = nPop - len(selected)
                selected.extend(front_sorted[:remaining])
                break

        return [combined[i] for i in selected]

    def _get_pareto_front(self, population):
        fronts = self.non_dominated_sort(population)
        if len(fronts) > 0:
            return [population[i] for i in fronts[0]]
        return []

    def _restart_population(self, restart_ratio=0.2, flip_ratio=0.1):
        n_restart = max(1, int(restart_ratio * self.nPop))
        nVar = self.dataFeatureNum
        pareto_pop = self._get_pareto_front(self.Position)
        if len(pareto_pop) == 0:
            pareto_pop = self.Position
        scores = np.array([
            ind['cost'] + 0 * ind['n_sel'] / nVar
            for ind in self.Position
        ])
        worst_indices = np.argsort(scores)[-n_restart:]

        for idx in worst_indices:
            ref = random.choice(pareto_pop)
            ref_mask = (ref['pos'] >= self.eta).astype(int)
            new_mask = ref_mask.copy()
            n_flip = max(1, int(flip_ratio * nVar))
            flip_idx = np.random.choice(nVar, size=n_flip, replace=False)
            new_mask[flip_idx] = 1 - new_mask[flip_idx]
            if np.sum(new_mask) == 0:
                if hasattr(self, 'fused_weights') and self.fused_weights is not None:
                    add_idx = np.argmax(self.fused_weights)
                else:
                    add_idx = np.random.randint(nVar)
                new_mask[add_idx] = 1
            new_pos = np.where(
                new_mask == 1,
                np.random.uniform(self.eta, 1.0, nVar),
                np.random.uniform(0.0, self.eta, nVar)
            )

            new_cost = self._cost_function(new_pos)
            new_n_sel = int(np.sum(new_mask))

            self.Position[idx] = {
                'pos': new_pos,
                'cost': new_cost,
                'n_sel': new_n_sel
            }

    def _evaluate_pareto_front(self, pareto_pop, is_test=False):
        records = []
        for idx, ind in enumerate(pareto_pop):
            if is_test:
                records.append({
                    'Individual_Index': idx,
                    'Error_Rate': ind['cost'],
                    'Accuracy': 1.0 - ind['cost'],
                    'N_Selected_Features': ind['n_sel']
                })
            else:
                records.append({
                    'Individual_Index': idx,
                    'Error_Rate': ind['cost'],
                    'Accuracy': 1.0 - ind['cost'],
                    'N_Selected_Features': ind['n_sel']
                })
        return records

    def local_search(self, pos, n_try=5):
        mask = (pos >= self.eta).astype(int)
        best_pos = pos.copy()
        best_cost = self._cost_function(best_pos)
        best_nsel = np.sum(mask)

        for _ in range(n_try):
            new_mask = mask.copy()

            selected = np.where(new_mask == 1)[0]
            unselected = np.where(new_mask == 0)[0]

            if len(selected) > 0:
                remove = selected[np.argmin(self.fused_weights[selected])]
                new_mask[remove] = 0

            if len(unselected) > 0 and np.random.rand() < 0.5:
                add = unselected[np.argmax(self.fused_weights[unselected])]
                new_mask[add] = 1

            new_pos = np.where(
                new_mask == 1,
                np.random.uniform(self.eta, 1.0, len(mask)),
                np.random.uniform(0.0, self.eta, len(mask))
            )

            new_cost = self._cost_function(new_pos)
            new_nsel = np.sum(new_mask)

            old = {'cost': best_cost, 'n_sel': best_nsel}
            new = {'cost': new_cost, 'n_sel': new_nsel}

            if self._dominates(new, old):
                best_pos = new_pos
                best_cost = new_cost
                best_nsel = new_nsel
                mask = new_mask

        return best_pos

    def run(self):
        nVar = self.dataFeatureNum
        VarMin = self.LB
        VarMax = self.UB

        pop_bin = self.mofno_init(
            self.nPop, self.dataFeatureNum,
            self.is_pareto_feature, self.is_intersection,
            self.fused_weights,
        )

        self.Position = []
        for i in range(self.nPop):
            pos = np.zeros(nVar)
            for j in range(nVar):
                if pop_bin[i][j] == 1:
                    pos[j] = np.random.uniform(self.eta, 1.0)
                else:
                    pos[j] = np.random.uniform(0.0, self.eta)
            cost = self._cost_function(pos)
            n_sel = int(np.sum(pop_bin[i]))
            self.Position.append({'pos': pos, 'cost': cost, 'n_sel': n_sel})

        best_idx = min(range(self.nPop), key=lambda i: self.Position[i]['cost'])
        self.BestPos = self.Position[best_idx]['pos'].copy()
        self.BestSolCost = self.Position[best_idx]['cost']

        self.BestCosts = np.zeros(self.MaxIt)
        self.BestCosts[0] = self.BestSolCost

        best_mask_0 = (self.BestPos >= self.eta).astype(int)
        n_sel_0 = np.sum(best_mask_0)
        self.generation_best_acc.append(1.0 - self.BestSolCost)
        self.generation_best_nsel.append(int(n_sel_0))

        stagnation = 0
        stagnation_limit = 10
        lambda_stag = 0
        best_mask_0 = (self.BestPos >= self.eta).astype(int)
        previous_score = self.BestSolCost + lambda_stag * np.sum(best_mask_0) / nVar
        positions = np.array([ind['pos'] for ind in self.Position])

        for it in range(1, self.MaxIt + 1):
            costs = np.array([ind['cost'] for ind in self.Position])
            sorted_idx = np.argsort(costs)
            DP = cdist(positions, positions, metric='euclidean')
            FB = np.zeros(self.nPop, dtype=int)
            NW = np.zeros(self.nPop, dtype=int)

            for i in range(self.nPop):
                rank = np.where(sorted_idx == i)[0][0]

                if rank == 0:
                    Betters = np.array([], dtype=int)
                    Worses = sorted_idx[rank + 1:]
                elif rank == self.nPop - 1:
                    Betters = sorted_idx[:rank]
                    Worses = np.array([], dtype=int)
                else:
                    Betters = sorted_idx[:rank]
                    Worses = sorted_idx[rank + 1:]

                if len(Betters) > 0:
                    keep_ratio = (1 - (it / self.MaxIt) ** 0.5) * np.random.rand()
                    keep_num = max(1, int(np.ceil(len(Betters) * keep_ratio)))
                    Betters = Betters[:keep_num]

                if len(Betters) > 0:
                    dists_to_betters = DP[Betters, i]
                    farthest_idx = np.argmax(dists_to_betters)
                    FB[i] = Betters[farthest_idx]
                else:
                    FB[i] = i

                if len(Worses) > 0:
                    dists_to_worses = DP[Worses, i]
                    nearest_idx = np.argmin(dists_to_worses)
                    NW[i] = Worses[nearest_idx]
                else:
                    NW[i] = i

            new_population = []
            for i in range(self.nPop):
                pos_i = self.Position[i]['pos']

                alpha = (1 - (it / self.MaxIt)) ** 0.5
                mu = np.random.rand()
                DFF = np.abs(np.random.uniform(mu - alpha, mu + alpha, nVar))
                W = (2 + np.random.uniform(-1, 1)) * np.random.randint(1, 3)

                diff_FB = self.Position[FB[i]]['pos'] - pos_i
                diff_NW = self.Position[NW[i]]['pos'] - pos_i
                S = np.maximum(0, np.sign(diff_FB) * np.sign(diff_NW))
                effective_S = np.where((S == 1) & (np.abs(diff_FB) < np.abs(diff_NW)), 0, S)

                step_abs = effective_S * np.abs(diff_NW) + (1 - effective_S) * np.abs(diff_FB)

                Pos_temp = (effective_S) * self.Position[NW[i]]['pos'] + (1 - effective_S) * pos_i + \
                           np.sign(diff_FB) * np.random.rand(nVar) * step_abs

                if np.random.rand() < 1 - (it / self.MaxIt):
                    new_pos = Pos_temp
                else:
                    Exp = DFF * (self.Position[FB[i]]['pos'] - Pos_temp)
                    new_pos = Pos_temp + W * Exp

                if np.random.rand() < 0.2:
                    new_pos = self.local_search(new_pos)
                new_pos = self._clipping(new_pos)
                mask = (new_pos >= self.eta).astype(int)
                flip_rate = 0.02 * (1 - it / self.MaxIt)
                flip_mask = np.random.rand(nVar) < flip_rate
                mask[flip_mask] = 1 - mask[flip_mask]

                new_pos = np.where(
                    mask == 1,
                    np.random.uniform(self.eta, 1.0, nVar),
                    np.random.uniform(0.0, self.eta, nVar)
                )
                new_cost = self._cost_function(new_pos)
                new_n_sel = np.sum((new_pos >= self.eta).astype(int))
                new_population.append({'pos': new_pos, 'cost': new_cost, 'n_sel': new_n_sel})

            combined = self.Position + new_population
            combined = self.delete_duplicate(combined)

            while len(combined) < self.nPop:
                pos = np.random.uniform(VarMin, VarMax, nVar)
                cost = self._cost_function(pos)
                n_sel = np.sum((pos >= self.eta).astype(int))
                combined.append({'pos': pos, 'cost': cost, 'n_sel': n_sel})

            self.Position = self.select_next_population(combined, self.nPop)
            positions = np.array([ind['pos'] for ind in self.Position])

            best_idx = min(range(len(self.Position)), key=lambda i: self.Position[i]['cost'])
            self.BestPos = self.Position[best_idx]['pos'].copy()
            self.BestSolCost = self.Position[best_idx]['cost']
            current_best = self.BestSolCost
            self.BestCosts[it - 1] = current_best

            best_mask = (self.BestPos >= self.eta).astype(int)
            n_sel = np.sum(best_mask)

            current_score = current_best + lambda_stag * n_sel / nVar
            if current_score < previous_score - 1e-12:
                stagnation = 0
                previous_score = current_score
            else:
                stagnation += 1
            if stagnation >= stagnation_limit:
                self._restart_population(
                    restart_ratio=0.5,
                    flip_ratio=0.1 * (1 - it / self.MaxIt) + 0.02
                )
                best_idx = min(range(len(self.Position)), key=lambda i: self.Position[i]['cost'])
                self.BestPos = self.Position[best_idx]['pos'].copy()
                self.BestSolCost = self.Position[best_idx]['cost']
                current_best = self.BestSolCost
                best_mask = (self.BestPos >= self.eta).astype(int)
                n_sel = np.sum(best_mask)
                previous_score = current_best + lambda_stag * n_sel / nVar
                stagnation = 0
            self.BestCosts[it - 1] = current_best
            self.generation_best_acc.append(1.0 - current_best)
            self.generation_best_nsel.append(int(n_sel))
            if it % 5 == 0:
                print(f"Iteration {it}: Best Cost = {current_best:.6f} len = {n_sel}")

        print("============================================")
        self.final_train_acc = 1.0 - self.BestSolCost
        print(f"train best acc = {self.final_train_acc:.6f}")
        best_mask = (self.BestPos >= self.eta).astype(int)
        self.final_n_sel = int(np.sum(best_mask))
        print(f"train best len = {self.final_n_sel}")
        print()

        if self.testX is not None and self.testY is not None:
            print("--- test ---")
            if self.final_n_sel > 0:
                knn = KNeighborsClassifier(n_neighbors=5, algorithm="auto", metric='manhattan')
                knn.fit(self.dataX[:, best_mask == 1], self.dataY)
                self.final_test_acc = knn.score(self.testX[:, best_mask == 1], self.testY)
                print(f"test acc = {self.final_test_acc:.6f}")
                print(f"features = {self.final_n_sel}")
            else:
                self.final_test_acc = None
            print()

        train_pareto = self._get_pareto_front(self.Position)
        self.pareto_front_history = self._evaluate_pareto_front(train_pareto, is_test=False)

        if self.testX is not None and self.testY is not None:
            test_population = []
            for ind in train_pareto:
                test_err, n_sel = self._test_cost_function(ind['pos'])
                test_population.append({'pos': ind['pos'], 'cost': test_err, 'n_sel': n_sel})
            test_pareto = self._get_pareto_front(test_population)
            self.test_pareto_front_history = self._evaluate_pareto_front(test_pareto, is_test=True)
        else:
            self.test_pareto_front_history = []

if __name__ == '__main__':
    ducName = []
    for ducName in ducName:
        path = "dataCSV/dataCSV/" + ducName + ".csv"
        dataCsv = ReadCSV(path=path)
        dataCsv.getData()
        N_RUNS = 30
        all_results = []
        seeds = list(range(0, N_RUNS))
        for run_id, seed in enumerate(seeds, 1):
            trainX, testX, trainY, testY = train_test_split(
                dataCsv.dataX, dataCsv.dataY,
                test_size=0.3,
                stratify=dataCsv.dataY,
                random_state = seed
            )
            run_start = time.time()
            mofno = MOFNO(dataX=trainX, dataY=trainY, dataName=ducName, testX=testX, testY=testY)
            mofno.setParameter(nPop=100, MaxIt=100, eta=0.6)
            mofno.run()
            runtime = time.time() - run_start
            test_acc_str = f"{mofno.final_test_acc:.4f}" if mofno.final_test_acc is not None else "N/A"
            print(f"No.{run_id} : Train={mofno.final_train_acc:.4f}, Test={test_acc_str}, "
                  f"Features={mofno.final_n_sel}, Time={runtime:.2f}s")