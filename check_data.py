import pickle
import os

def check():
    if not os.path.exists("simulation_state.pkl"):
        print("No hay archivo.")
        return
    with open("simulation_state.pkl", "rb") as f:
        data = pickle.load(f)
    
    print(f"--- CONTENIDO DEL PKL ---")
    print(f"Claves encontradas: {list(data.keys())}")
    if 'ports' in data:
        print(f"Puertos guardados: {len(data['ports'])}")
        for p in data['ports']:
            print(f" -> Puerto ID: {p['id']} en Pos: {p['pos']}")
    else:
        print("❌ NO HAY PUERTOS EN EL ARCHIVO")

check()