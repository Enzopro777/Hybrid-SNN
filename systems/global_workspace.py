
#systems/Global_workspace.py

class GlobalWorkspace:
    def update(self, net):
        best_r = None
        best_score = -999

        # Buscar la región más dominante
        for i, r in enumerate(net.regions):
            score = r.dopamine + r.energy

            if score > best_score:
                best_score = score
                best_r = i

        net.global_workspace["region"] = best_r
        net.global_workspace["signal"] = best_score