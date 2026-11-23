import csv
import sys
import math

X = []
Y = []

# Read CSV
try:
    with open("data.csv") as f:
        data = csv.DictReader(f)
        if data.fieldnames is None or "km" not in data.fieldnames or "price" not in data.fieldnames:
            print("Error: file must contain 'km' and 'price' columns", file=sys.stderr)
            sys.exit(1)
        for line in data:
            try:
                x = float(line["km"])
                y = float(line["price"])
            except (ValueError, TypeError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            X.append(x)
            Y.append(y)
except FileNotFoundError:
    print("Error: data.csv not found", file=sys.stderr)
    sys.exit(1)
except UnicodeDecodeError:
    print("Error: data.csv is not a valid text file", file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(f"Error: cannot read data.csv: {e}", file=sys.stderr)
    sys.exit(1)

n = len(X)
if n == 0:
    print("Error: no valid rows to evaluate", file=sys.stderr)
    sys.exit(1)

# Read thetas (0 if the model has not been trained yet)
theta0 = theta1 = 0.0
values = None
try:
    with open("thetas.txt") as f:
        values = f.read().strip().split(",")
except FileNotFoundError:
    print("Warning: model not trained yet, using theta0 = theta1 = 0", file=sys.stderr)
except UnicodeDecodeError:
    print("Error: thetas.txt is not a valid text file", file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(f"Error: cannot read thetas.txt: {e}", file=sys.stderr)
    sys.exit(1)

if values is not None:
    try:
        if len(values) != 2:
            raise ValueError
        theta0 = float(values[0])
        theta1 = float(values[1])
        if not (math.isfinite(theta0) and math.isfinite(theta1)):
            raise ValueError
    except ValueError:
        print("Error: thetas.txt is corrupted, run train.py again", file=sys.stderr)
        sys.exit(1)


def estimate_price(mileage):
    return theta0 + theta1 * mileage


# Compare each prediction to the real price
mean_price = sum(Y) / n
sum_abs_err = 0
sum_sq_err = 0
sum_sq_total = 0
for i in range(n):
    err = estimate_price(X[i]) - Y[i]
    sum_abs_err += abs(err)
    sum_sq_err += err ** 2
    sum_sq_total += (Y[i] - mean_price) ** 2

# MAE (Mean Absolute Error): average gap between predicted and real price,
# in euros, ignoring the sign. "On average, the model is wrong by X euros."
mae = sum_abs_err / n

# RMSE (Root Mean Squared Error): like MAE, in euros, but errors are squared
# before averaging, so big mistakes weigh more. Same formula as a standard
# deviation: it is the standard deviation of the errors.
rmse = math.sqrt(sum_sq_err / n)

print(f"Evaluated on {n} rows")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")

# R2 (coefficient of determination): share of the price variation explained
# by the mileage. R2 = 1 - sum_sq_err / sum_sq_total, where sum_sq_total is the
# error of a naive model that always predicts the mean price.
# 1 = perfect, 0 = no better than the mean, negative = worse than the mean.
if sum_sq_total == 0:
    print("R2   = not defined (all prices are identical)")
else:
    r2 = 1 - sum_sq_err / sum_sq_total
    print(f"R2   = {r2:.4f}")
