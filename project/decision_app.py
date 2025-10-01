
from dataclasses import dataclass
import argparse
import pandas as pd

@dataclass
class Trapezoid:
    a: float
    b: float
    c: float
    d: float

    def __add__(self, other):
        return Trapezoid(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def scalar_mul(self, scalar: float):
        return Trapezoid(self.a * scalar, self.b * scalar, self.c * scalar, self.d * scalar)

    def centroid(self):
        return (self.a + 2*self.b + 2*self.c + self.d) / 6.0

    def to_list(self):
        return [self.a, self.b, self.c, self.d]

    def __repr__(self):
        return f"Trapezoid({self.a:.3f},{self.b:.3f},{self.c:.3f},{self.d:.3f})"


LINGUISTIC_SCALE = {
    "VL": Trapezoid(0.0, 0.0, 0.05, 0.2),
    "L":  Trapezoid(0.05, 0.2, 0.2, 0.35),
    "ML": Trapezoid(0.2, 0.35, 0.35, 0.5),
    "M":  Trapezoid(0.35, 0.5, 0.5, 0.65),
    "MH": Trapezoid(0.5, 0.65, 0.65, 0.8),
    "H":  Trapezoid(0.65, 0.8, 0.8, 0.95),
    "VH": Trapezoid(0.8, 0.95, 1.0, 1.0)
}


def fuzzy_weighted_average(trapezoids, weights):
    agg = Trapezoid(0.0, 0.0, 0.0, 0.0)
    for t, w in zip(trapezoids, weights):
        agg = agg + t.scalar_mul(w)
    return agg

def defuzz_pessimistic(trap: Trapezoid):
    return trap.a

def defuzz_optimistic(trap: Trapezoid):
    return trap.d

def defuzz_neutral(trap: Trapezoid):
    return trap.centroid()

def compute_scores(alternatives, criteria, weights_dict, ratings, scale=LINGUISTIC_SCALE):
    results = []
    weights_list = [weights_dict[c] for c in criteria]
    for alt in alternatives:
        traps = [scale[ratings[alt][c]] for c in criteria]
        agg = fuzzy_weighted_average(traps, weights_list)
        pess = defuzz_pessimistic(agg)
        opti = defuzz_optimistic(agg)
        neut = defuzz_neutral(agg)
        results.append({
            "Alternative": alt,
            "a": agg.a, "b": agg.b, "c": agg.c, "d": agg.d,
            "Pessimistic": pess,
            "Optimistic": opti,
            "Neutral": neut
        })
    df = pd.DataFrame(results).set_index("Alternative")
    df["Pess_rank"] = df["Pessimistic"].rank(ascending=False, method="min").astype(int)
    df["Opt_rank"] = df["Optimistic"].rank(ascending=False, method="min").astype(int)
    df["Neut_rank"] = df["Neutral"].rank(ascending=False, method="min").astype(int)
    return df

def save_results_csv(df, path="results.csv"):
    df.to_csv(path)
    print(f"Results saved to {path}")

def demo_example():
    alternatives = ["A1","A2","A3","A4","A5"]
    criteria = ["C1","C2","C3","C4","C5","C6","C7"]
    weights = {"C1":0.15,"C2":0.10,"C3":0.15,"C4":0.10,"C5":0.15,"C6":0.20,"C7":0.15}
    ratings = {
        "A1": {"C1":"MH","C2":"M","C3":"H","C4":"ML","C5":"M","C6":"H","C7":"MH"},
        "A2": {"C1":"M","C2":"ML","C3":"MH","C4":"M","C5":"MH","C6":"MH","C7":"M"},
        "A3": {"C1":"L","C2":"L","C3":"M","C4":"L","C5":"ML","C6":"M","C7":"L"},
        "A4": {"C1":"VH","C2":"H","C3":"VH","C4":"MH","C5":"H","C6":"VH","C7":"VH"},
        "A5": {"C1":"ML","C2":"M","C3":"ML","C4":"ML","C5":"M","C6":"ML","C7":"ML"}
    }
    df = compute_scores(alternatives, criteria, weights, ratings)
    print(df)
    save_results_csv(df, "results_demo.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision support with trapezoidal linguistic terms")
    parser.add_argument("--demo", action="store_true", help="Run demo example and save results_demo.csv")
    args = parser.parse_args()
    if args.demo:
        demo_example()
    else:
        demo_example()
