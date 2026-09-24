#system/prediction.py

class PredictionSystem:
    def __init__(self):
        self.prev_energy = 0.0

    def update(self, net):
        # Valor actual
        current = net.self_state.get("energy", 0.0)

        # 🔮 Predicción = último valor conocido
        predicted = self.prev_energy

        # ❌ Error real
        error = abs(current - predicted)

        net.self_state["prediction_error"] = error

        # 🧠 Guardar para el próximo paso
        self.prev_energy = current