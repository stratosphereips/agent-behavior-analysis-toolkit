import csv, glob, os

rows = [["env", "model", "seed", "path"]]
for f in sorted(glob.glob('results/frozenlake8x8/random/**/*_20_metrics.json', recursive=True)):
    seed = os.path.basename(f).split('_')[1]
    rows.append(["FrozenLake", "random", seed, f.replace(os.sep, '/')])
for f in sorted(glob.glob('results/mountain_car/random/seed[1-5]/*_20_metrics.json')):
    seed = os.path.basename(os.path.dirname(f)).replace('seed', '')
    rows.append(["MountainCar", "random", seed, f.replace(os.sep, '/')])
rows.append(["MountainCar", "random", "4242",
             "results/mountain_car/random/seed4242/seed_4242_50_metrics.json"])
csv.writer(open('scripts/_random_floor.csv', 'w', newline='')).writerows(rows)
print(f"wrote {len(rows)-1} random rows")
