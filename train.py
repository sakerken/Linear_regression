import csv
import sys
import math

LEARNING_RATE = 0.1
ITERATIONS = 20000

X = []
Y = []

# Read CSV
try :
    with open("data.csv") as f:
        data = csv.DictReader(f)
        if data.fieldnames is None or "km" not in data.fieldnames or "price" not in data.fieldnames:
            print("Error: file must contain 'km' and 'price' columns", file=sys.stderr)
            sys.exit(1)
        for line in data:
            try :
                x = float(line["km"])
                y = float(line["price"])
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            X.append(x)
            Y.append(y)
except FileNotFoundError:
    print("Error: file not found", file=sys.stderr)
    sys.exit(1)
except UnicodeDecodeError:
    print("Error: file is not a valid text file", file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(f"Error: cannot read file: {e}", file=sys.stderr)
    sys.exit(1)

# Normalise
n = len(X)
if n < 2:
    print("Error: at least 2 valid rows are needed to train the model", file=sys.stderr)
    sys.exit(1)
mean = sum(X) / n
std = math.sqrt(sum((x - mean) ** 2 for x in X) / n)
if std == 0:
    print("Error: all mileages are identical, cannot fit a line", file=sys.stderr)
    sys.exit(1)
X_norm = [(x - mean) / std for x in X]

# Gradient Descent
a = b = 0
def estimate_price(x):
    return a * x + b

for _ in range(ITERATIONS):
    erra = errb = 0
    for i in range(len(X_norm)):
        estp = estimate_price(X_norm[i])
        err = estp - Y[i]
        errb += err
        erra += err * X_norm[i]
    tmp_b = LEARNING_RATE * errb / n
    tmp_a = LEARNING_RATE * erra / n
    a -= tmp_a
    b -= tmp_b

# denormalise
theta1 = a / std
theta0 = b - a * mean / std

# Save
try:
    with open("thetas.txt", "w") as f:
        f.write(f"{theta0},{theta1}\n")
except OSError as e:
    print(f"Error: cannot write file: {e}", file=sys.stderr)
    sys.exit(1)

print(f"theta0 = {theta0}")
print(f"theta1 = {theta1}")


# Bonus: plot the data and the regression line (python3 train.py --plot)
if "--plot" in sys.argv:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed, skipping plot", file=sys.stderr)
    else:
        km_min = min(X)
        km_max = max(X)
        plt.scatter(X, Y, label="Data")
        plt.plot([km_min, km_max],
                 [theta0 + theta1 * km_min, theta0 + theta1 * km_max],
                 color="red", label="Linear regression")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price")
        plt.title("Car price by mileage")
        plt.legend()
        try:
            plt.savefig("plot.png")
        except OSError as e:
            print(f"Warning: cannot save plot.png: {e}", file=sys.stderr)
        plt.show()

