import spider
from sentence_transformers import SentenceTransformer
import os
import shutil

# --- Config ---
DB_PATH = "./spider_graph.db"
MODEL_NAME = 'all-MiniLM-L6-v2'  # Free, local, fast

def main():
    # 1. Clean Slate
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    print("🕸️  Initializing Spider DB...")
    encoder = SentenceTransformer(MODEL_NAME)
    
    # Initialize Rust DB (parameters matching your src/db.rs defaults)
    db = spider.SpiderDB(DB_PATH, max_capacity=10000, m=16, ef_construction=200)

    # 2. Ingest Data (Mixed topics to test clustering)
    knowledge = [
        # === PHYSICS (Quantum & General) ===
        ("Quantum entanglement allows instantaneous correlation between particles across vast distances", 9),
        ("Wave-particle duality suggests light behaves as both a wave and a stream of particles", 9),
        ("Heisenberg's uncertainty principle states you cannot know position and momentum simultaneously", 8),
        ("Schrödinger's equation is the fundamental equation of physics for describing quantum mechanical behavior", 8),
        ("String theory attempts to unify quantum mechanics and gravity using one-dimensional strings", 7),
        ("The observer effect implies that the act of measuring a quantum system affects its state", 8),
        ("Quantum superposition allows a particle to exist in multiple states at once until observed", 9),
        ("General relativity explains gravity as the curvature of spacetime caused by mass", 9),
        ("Dark matter makes up about 85% of the matter in the universe but does not interact with light", 8),
        ("The second law of thermodynamics states that entropy in an isolated system always increases", 8),
        ("Neutron stars are the collapsed cores of massive stars and are incredibly dense", 7),
        ("Hawking radiation suggests that black holes eventually evaporate over time", 7),

        # === PROGRAMMING (Rust, Python, Concepts) ===
        ("Python uses dynamic typing which allows for rapid prototyping but can lead to runtime errors", 8),
        ("Rust ensures memory safety without garbage collection through its ownership model", 9),
        ("JavaScript async/await is syntactic sugar over Promises for handling asynchronous operations", 7),
        ("Static type systems catch type-related bugs at compile time rather than runtime", 8),
        ("Functional programming emphasizes immutability and pure functions to reduce side effects", 7),
        ("The borrow checker in Rust prevents data races by enforcing strict borrowing rules", 9),
        ("Object-oriented programming organizes software design around data, or objects, rather than functions", 6),
        ("Concurrency allows multiple computations to happen simultaneously, improving performance", 7),
        ("Recursion is a method where the solution to a problem depends on solutions to smaller instances", 6),
        ("Docker containers package software with all its dependencies for consistent deployment", 8),
        ("Git is a distributed version control system for tracking changes in source code", 9),
        ("REST APIs use standard HTTP methods like GET, POST, PUT, and DELETE", 7),

        # === COOKING (Science & Technique) ===
        ("The Maillard reaction is a chemical reaction between amino acids and reducing sugars that gives browned food its flavor", 8),
        ("Sous vide involves cooking food in vacuum-sealed bags at precise temperatures for consistent results", 7),
        ("Fermentation is a metabolic process that produces chemical changes in organic substrates through enzymes", 7),
        ("Knife skills, such as the claw grip, are fundamental to safety and efficiency in the kitchen", 6),
        ("Emulsification is the process of mixing two liquids that are normally unmixable, like oil and water", 7),
        ("Caramelization is the oxidation of sugar, a process used extensively in cooking for the resulting nutty flavor and brown color", 6),
        ("Resting meat after cooking allows the juices to redistribute throughout the fibers", 8),
        ("Umami is the fifth basic taste, characterized by savory flavor found in broths and cooked meats", 7),
        ("Proofing yeast is the process of checking if yeast is alive and active before baking", 6),
        ("Tempering chocolate involves heating and cooling it to stabilize the cocoa butter crystals", 7),
        ("Mise en place is the philosophy of setting up all ingredients and tools before cooking begins", 8),

        # === MUSIC (Theory & Production) ===
        ("Modal jazz explores scales (modes) beyond the traditional major and minor keys", 7),
        ("The circle of fifths is a visual representation of the relationships among the 12 tones of the chromatic scale", 8),
        ("Polyrhythms involve the simultaneous use of two or more conflicting rhythms", 7),
        ("Harmonic overtones are frequencies that sound above the fundamental note, giving instruments their timbre", 8),
        ("Counterpoint is the relationship between voices that are harmonically interdependent yet independent in rhythm", 7),
        ("Syncopation involves stressing the weak beats rather than the strong beats", 6),
        ("A synthesizer generates audio signals to create sounds not possible with acoustic instruments", 7),
        ("Compression in audio engineering reduces the dynamic range of a recording", 7),
        ("The pentatonic scale consists of five notes per octave and is ubiquitous in world music", 6),
        ("Reverb simulates the reflection of sound waves in a physical space", 6),

        # === BIOLOGY (Genetics & Cellular) ===
        ("DNA replication ensures genetic continuity by creating an identical copy of a DNA molecule", 9),
        ("Mitochondria are known as the powerhouse of the cell because they generate most of the cell's supply of ATP", 9),
        ("Natural selection is the differential survival and reproduction of individuals due to differences in phenotype", 9),
        ("Photosynthesis is the process used by plants to convert light energy into chemical energy", 8),
        ("CRISPR is a genetic engineering tool that uses a bacterial immune system to edit DNA", 8),
        ("Mitosis is the part of the cell cycle when replicated chromosomes are separated into two new nuclei", 7),
        ("Enzymes are biological catalysts that speed up chemical reactions in living organisms", 7),
        ("Homeostasis is the state of steady internal, physical, and chemical conditions maintained by living systems", 8),
        ("Neurons communicate with each other via electrical events called action potentials", 7),
        ("The ribosome is a complex molecular machine found within all living cells that serves as the site of protein synthesis", 8),

        # === HISTORY (New Cluster) ===
        (" The Industrial Revolution marked a major turning point in history with the rise of factories and steam power", 8),
        ("The fall of the Roman Empire in 476 AD led to the beginning of the Middle Ages in Europe", 7),
        ("The Silk Road was a network of trade routes connecting the East and West", 7),
        ("The printing press, invented by Gutenberg, revolutionized the spread of information", 9),
        ("The Cold War was a period of geopolitical tension between the Soviet Union and the United States", 8),
        ("The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth", 8),

        # === BRIDGE CONCEPTS (Connecting Clusters) ===
        ("Quantum computing uses phenomena like superposition to solve problems faster than classical computers", 9), # Physics <-> Programming
        ("Molecular gastronomy applies scientific principles from chemistry and physics to cooking", 8), # Physics <-> Cooking
        ("Bioinformatics combines biology, computer science, and statistics to analyze genetic data", 8), # Biology <-> Programming
        ("Psychoacoustics is the scientific study of sound perception and audiology", 7), # Physics <-> Music
        ("Algorithmic composition uses computer programs to create music automatically", 7), # Programming <-> Music
    ]

    print("\n--- Ingesting Nodes ---")
    for text, sig in knowledge:
        emb = encoder.encode(text).tolist()
        # auto_link_threshold=0.4 ensures strictly relevant connections
        node_id = db.add_node(text, emb, sig, 0.4) 
        print(f"Added Node {node_id}: {text[:30]}...")

    # 3. Test Internal Graph Logic
    print("\n--- Testing Graph & Clustering ---")
    
    # Force cluster build (Calls src/cluster.rs logic)
    db.build_clusters(k_clusters=3)
    print("Clusters built.")
    
    # 4. Save to Disk
    db.save(None)
    print(f"✅ Database saved to {DB_PATH}")

if __name__ == "__main__":
    main()