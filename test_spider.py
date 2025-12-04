import spider
import time

def test_modular_architecture():
    print("--- Testing Modular Architecture ---")
    db = spider.SpiderDB()
    
    # 1. Add Nodes
    id1 = db.add_node("Node A", [0.1]*384, 10)
    id2 = db.add_node("Node B", [0.1]*384, 5)
    id3 = db.add_node("Node C", [0.1]*384, 1)
    print(f"Added nodes: {id1}, {id2}, {id3}")
    
    # 2. Add Edge (Mocked)
    db.add_edge(id1, id2)
    print("Added edge from Node A to Node B")
    
    # 3. Hybrid Search
    # Create a dummy query vector (size 384)
    query = [0.1] * 384
    results = db.hybrid_search(query, 2)
    print(f"Hybrid Search Results: {results}")
    
    # 4. Vacuum
    dead = db.vacuum(5.0)
    print(f"Vacuum (Threshold 5.0): {dead}")
    
    # Node C (Sig 1) should be dead. Node A (Sig 10) and B (Sig 5) should live.
    # Note: Time delta is 0, so score is roughly (Sig * 10) / 3.48
    # A: 100 / 3.48 = 28.7
    # B: 50 / 3.48 = 14.3
    # C: 10 / 3.48 = 2.87
    
    assert id3 in dead, "Node C should be dead"
    assert id1 not in dead, "Node A should be alive"
    assert id2 not in dead, "Node B should be alive"
    
    print("SUCCESS: Modular architecture verified!")

if __name__ == "__main__":
    test_modular_architecture()
