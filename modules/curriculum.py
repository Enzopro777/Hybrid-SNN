# modules/curriculum.py
"""
Currícula adaptativa de vocabulario — Etapa 2 del roadmap de letras.

Hasta ahora el sistema sabía entrenar exactamente 2 símbolos (X, O) porque
todo el cableado en LetterIOModule.auto_connect() estaba escrito a mano
para un split binario. Este módulo NO toca neuronas ni puertos: es un
coordinador puro que decide QUÉ LETRA entrenar en cada trial, a partir del
desempeño reciente de la red en cada letra ya desbloqueada.

Idea central (currícula adaptativa real, no solo "más símbolos"):
  1. Arranca con las letras que ya sabemos que andan (X, O — validadas en
     Fase A) más el resto del vocabulario objetivo bloqueado.
  2. Cada trial reporta evidencia del readout, éxito y margen de separación a record_trial().
  3. Cuando TODAS las letras activas llevan suficientes trials Y están
     "dominadas" (evidencia y tasa de éxito por encima de un umbral en su
     ventana reciente), se desbloquea la siguiente letra del vocabulario.
  4. next_letter() no elige uniforme al azar entre las letras activas:
     les da más peso a las que peor van, para no dejar de practicarlas
     mientras el resto sigue subiendo (repaso espaciado simple).

Persistencia: este objeto no es un core.module.Module (no tiene puertos),
así que no pasa por el mecanismo de guardado de puertos. En su lugar,
SimulationEngine guarda/restaura su estado como un dict plano en
`net.curriculum_state`, que el guardado híbrido ya persiste gratis (es un
atributo picklable cualquiera de `net`).
"""

import random


class VocabularyCurriculum:
    def __init__(self, vocabulary, starter_letters=None, window=8,
                 mastery_evidence=0.5, mastery_success_rate=0.7,
                 mastery_margin=0.0,
                 min_trials_to_unlock=6, weak_letter_boost=3.0,
                 balanced_mode=False, max_trial_imbalance=1):
        """
        vocabulary            : vocabulario objetivo completo, en el orden en
                                 que se van desbloqueando (ej: ["X","O","T","A","E"]).
        starter_letters        : letras que arrancan ya desbloqueadas. Si es
                                 None, arranca solo con vocabulary[0].
        window                 : cuántos trials recientes por letra se usan
                                 para juzgar el dominio actual (media móvil).
        mastery_evidence       : evidencia_real promedio mínima (0-1) para
                                 considerar dominada una letra dentro de su
                                 ventana.
        mastery_success_rate   : fracción mínima de trials correctamente
                                 clasificados dentro de la ventana.
        min_trials_to_unlock   : no se evalúa ningún desbloqueo hasta que
                                 TODAS las letras activas tengan al menos
                                 esta cantidad de trials acumulados.
        mastery_margin        : separación media mínima entre el score del
                                 pool ganador y el segundo pool del learned readout.
        weak_letter_boost      : cuánto peso extra reciben, al elegir la
                                 próxima letra, las letras con peor evidencia
                                 promedio respecto al umbral de dominio.
        """
        self.vocabulary = list(vocabulary)
        starters = list(starter_letters) if starter_letters else [self.vocabulary[0]]
        self.window = window
        self.mastery_evidence = mastery_evidence
        self.mastery_success_rate = mastery_success_rate
        self.mastery_margin = float(max(0.0, mastery_margin))
        self.min_trials_to_unlock = min_trials_to_unlock
        self.weak_letter_boost = weak_letter_boost
        # Por compatibilidad, las instancias históricas fuera del currículo
        # principal siguen usando selección ponderada salvo que el caller active
        # explícitamente el modo balanceado (1.6.13 lo hace desde config.py).
        self.balanced_mode = bool(balanced_mode)
        self.max_trial_imbalance = max(0, int(max_trial_imbalance))

        self.history = {sym: [] for sym in self.vocabulary}
        self.total_trials = {sym: 0 for sym in self.vocabulary}
        self.unlocked = [s for s in self.vocabulary if s in starters]
        if not self.unlocked:
            self.unlocked = [self.vocabulary[0]]

    # ------------------------------------------------------------
    # Serialización — dict plano, 100% picklable
    # ------------------------------------------------------------
    def to_dict(self):
        return {
            "vocabulary": self.vocabulary,
            "unlocked": self.unlocked,
            "history": self.history,
            "total_trials": self.total_trials,
            "balanced_mode": self.balanced_mode,
            "max_trial_imbalance": self.max_trial_imbalance,
            "mastery_margin": self.mastery_margin,
        }

    def load_dict(self, data):
        if not data:
            return
        self.vocabulary = data.get("vocabulary", self.vocabulary)
        self.unlocked = data.get("unlocked", self.unlocked)
        self.history = data.get("history", self.history)
        self.total_trials = data.get("total_trials", self.total_trials)
        self.balanced_mode = bool(data.get("balanced_mode", self.balanced_mode))
        self.max_trial_imbalance = max(0, int(data.get("max_trial_imbalance", self.max_trial_imbalance)))
        self.mastery_margin = float(max(0.0, data.get("mastery_margin", self.mastery_margin)))
        # Si el vocabulario objetivo creció desde el último guardado
        # (Enzo agregó una letra nueva en config.py), completamos lo que falte
        # sin pisar lo que ya había.
        for sym in self.vocabulary:
            self.history.setdefault(sym, [])
            self.total_trials.setdefault(sym, 0)

    # ------------------------------------------------------------
    # Registro de resultados de un trial
    # ------------------------------------------------------------
    def record_trial(self, letter, evidencia_real, exito, complete=True, margin=0.0):
        # v0.8: un trial abortado/incompleto no cuenta como evidencia de
        # aprendizaje ni consume el cupo mínimo para desbloqueo.
        if not complete:
            return False
        if letter not in self.history:
            self.history[letter] = []
            self.total_trials[letter] = 0
        self.history[letter].append((float(evidencia_real), bool(exito), float(max(0.0, margin))))
        self.history[letter] = self.history[letter][-self.window:]
        self.total_trials[letter] += 1
        self._maybe_unlock_next()
        return True

    # ------------------------------------------------------------
    # Estado de dominio por letra
    # ------------------------------------------------------------
    def mastery(self, letter):
        """(evidencia_promedio, tasa_de_exito) sobre la ventana reciente."""
        hist = self.history.get(letter, [])
        if not hist:
            return 0.0, 0.0
        avg_evidence = sum(float(item[0]) for item in hist) / len(hist)
        success_rate = sum(1 for item in hist if bool(item[1])) / len(hist)
        return avg_evidence, success_rate

    def mastery_detail(self, letter):
        """(evidencia, éxito, margen) sobre la ventana reciente.

        El tercer campo es el margen de separación del learned readout
        (certeza top-vs-second). Historiales 1.6.13 de dos campos se aceptan
        y se consideran con margen 0 para no inventar evidencia retrospectiva.
        """
        hist = self.history.get(letter, [])
        if not hist:
            return 0.0, 0.0, 0.0
        avg_evidence = sum(float(item[0]) for item in hist) / len(hist)
        success_rate = sum(1 for item in hist if bool(item[1])) / len(hist)
        margins = [float(item[2]) for item in hist if len(item) >= 3]
        avg_margin = sum(margins) / len(margins) if margins else 0.0
        return avg_evidence, success_rate, avg_margin

    def is_mastered(self, letter):
        avg_evidence, success_rate, avg_margin = self.mastery_detail(letter)
        return (avg_evidence >= self.mastery_evidence and
                success_rate >= self.mastery_success_rate and
                avg_margin >= self.mastery_margin)

    def _maybe_unlock_next(self):
        locked = [s for s in self.vocabulary if s not in self.unlocked]
        if not locked:
            return
        if any(self.total_trials[s] < self.min_trials_to_unlock for s in self.unlocked):
            return
        if all(self.is_mastered(s) for s in self.unlocked):
            nueva = locked[0]
            self.unlocked.append(nueva)
            print(f"🎓 [Currícula] '{nueva}' desbloqueada — "
                  f"vocabulario activo: {self.unlocked}", flush=True)

    # ------------------------------------------------------------
    # Selección de la próxima letra a entrenar
    # ------------------------------------------------------------
    def next_letter(self):
        """Elige una letra activa, con más chance para las que van peor.
        Devuelve None si todavía no hay ninguna letra desbloqueada."""
        if not self.unlocked:
            return None

        # v1.6.13: modo balanceado. La prioridad es que ninguna letra se
        # quede atrás por azar: primero elegimos entre las clases con menos
        # trials; solo dentro de ese grupo usamos la debilidad/evidencia para
        # romper empates. Así, con 5 letras activas, las cuentas permanecen
        # dentro del desbalance máximo configurado.
        if self.balanced_mode:
            counts = {s: int(self.total_trials.get(s, 0)) for s in self.unlocked}
            min_count = min(counts.values())
            candidatos = [s for s in self.unlocked if counts[s] <= min_count + self.max_trial_imbalance - 1]
            if candidatos:
                candidatos.sort(key=lambda s: (counts[s], self.mastery(s)[0], s))
                lowest = counts[candidatos[0]]
                tier = [s for s in candidatos if counts[s] == lowest]
                if len(tier) > 1:
                    # Priorizar la más débil y desempatar de forma estable/aleatoria.
                    weak_score = {s: self.mastery_evidence - self.mastery(s)[0] for s in tier}
                    max_weak = max(weak_score.values())
                    weak_tier = [s for s in tier if abs(weak_score[s] - max_weak) < 1e-9]
                    return random.choice(weak_tier)
                return tier[0]

        pesos = []
        min_count = min(self.total_trials.get(s, 0) for s in self.unlocked)
        for sym in self.unlocked:
            avg_evidence, success_rate = self.mastery(sym)
            debilidad = max(0.0, self.mastery_evidence - avg_evidence)
            deficit = max(0, min_count + 2 - self.total_trials.get(sym, 0))
            success_gap = max(0.0, self.mastery_success_rate - success_rate)
            pesos.append(
                (1.0 + debilidad * self.weak_letter_boost + success_gap)
                * (1.0 + 0.35 * deficit)
            )
        return random.choices(self.unlocked, weights=pesos, k=1)[0]

    # ------------------------------------------------------------
    # Resumen legible para prints/heartbeat
    # ------------------------------------------------------------
    def status(self):
      partes = []
      for sym in self.vocabulary:
          if sym in self.unlocked:
              avg_evidence, success_rate, avg_margin = self.mastery_detail(sym)
              icono = "✅" if self.is_mastered(sym) else "📚"
              partes.append(
                  f"{sym}{icono}(ev={avg_evidence:.2f},ok={success_rate:.0%},"
                  f"m={avg_margin:.2f},n={self.total_trials[sym]})"
              )
          else:
              partes.append(f"{sym}🔒")

      resumen = " | ".join(partes)

      # --- NUEVO: por qué no se desbloqueó la siguiente todavía ---
      locked = [s for s in self.vocabulary if s not in self.unlocked]
      if locked:
          faltan_trials = {
              s: self.min_trials_to_unlock - self.total_trials[s]
              for s in self.unlocked
              if self.total_trials[s] < self.min_trials_to_unlock
          }
          if faltan_trials:
              detalle = ", ".join(f"{s}:+{n}" for s, n in faltan_trials.items())
              resumen += f" | ⏳ faltan trials para evaluar desbloqueo ({detalle})"
          elif not all(self.is_mastered(s) for s in self.unlocked):
              sin_dominar = [s for s in self.unlocked if not self.is_mastered(s)]
              resumen += f" | ⏳ esperando dominio de {', '.join(sin_dominar)} para desbloquear '{locked[0]}'"

      return resumen