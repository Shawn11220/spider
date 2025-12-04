import spider
import time

print("🕷️  Initializing Spider Brain...")
db = spider.SpiderDB()

# 1. Add a "Trivial" Node (Meme)
# Significance = 1 (Low), Dummy Vector = [0.0, 0.0]
meme_id = db.add_node("Funny Cat Meme", [0.0, 0.0], 1)
print(f"➕ Added Meme (ID: {meme_id}) | Significance: 1")

# 2. Add a "Critical" Node (Password)
# Significance = 100 (High)
pass_id = db.add_node("Bank Password: 123", [1.0, 1.0], 100)
print(f"➕ Added Password (ID: {pass_id}) | Significance: 100")

# 3. Simulate Usage
# We read the Password (Reinforcing it)
print("\n👀 Reading Password Node...")
content = db.get_node(pass_id)
print(f"   Retrieved: {content}")

# We DO NOT read the Meme (It starts decaying immediately)

# 4. Check Life Scores
print("\n📊 Calculating Life Scores...")
meme_score = db.calculate_life_score(meme_id)
pass_score = db.calculate_life_score(pass_id)

print(f"   Meme Score: {meme_score:.4f} (Should be low)")
print(f"   Pass Score: {pass_score:.4f} (Should be high)")

# 5. Run Vacuum (Pruning)
# Threshold is set to 5.0. 
# The Meme (Score ~3.0) should die. The Password (Score ~200) should live.
print("\n🧹 Running Vacuum (Threshold: 5.0)...")
dead_nodes = db.vacuum(5.0)

if meme_id in dead_nodes:
    print("✅ SUCCESS: The Meme was correctly pruned.")
else:
    print("❌ FAILURE: The Meme survived (Check logic).")

if pass_id not in dead_nodes:
    print("✅ SUCCESS: The Password survived.")
else:
    print("❌ FAILURE: The Password was deleted (Check logic).")