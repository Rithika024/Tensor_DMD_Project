from tt_dmd import load_climate_tensor, run_tt_dmd, DATA_PATH

video = load_climate_tensor(DATA_PATH)
tt_rec, eig, tt_err, details = run_tt_dmd(video)

print("\n=== DETAILS STRUCTURE ===")
print("Type:", type(details))
print("Length:", len(details) if isinstance(details, tuple) else "not a tuple")

if isinstance(details, tuple):
    for i, item in enumerate(details):
        print(f"\nItem {i}:")
        print("  Type:", type(item))
        if hasattr(item, "shape"):
            print("  Shape:", item.shape)
        else:
            print("  Value:", repr(item)[:200])
