import sys
import math

# Read thetas
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


# Estimate Price
def estimate_price(mileage):
    return theta0 + theta1 * mileage


# User Input
while True:
    try:
        text = input("Enter a mileage (km): ")
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)
    try:
        mileage = float(text)
    except ValueError:
        print("Error: please enter a number", file=sys.stderr)
        continue
    if not math.isfinite(mileage) or mileage < 0:
        print("Error: mileage must be a positive number", file=sys.stderr)
        continue
    break

price = estimate_price(mileage)
print(f"Estimated price: {price:.2f}")
if price < 0:
    print("Warning: negative estimate, this mileage is beyond what the model can handle", file=sys.stderr)
