# systems/genetics.py
from core.evolution_config import EVO_MENU, PROFILE
import random
import numpy as np
from config import GRID_SIZE  # Importación necesaria para normalizar correctamente

class GeneticSystem:
    def __init__(self):
        self.mutation_rate = 0.05
        self.crossover_prob = 0.7
        # NUEVO: Fuerza del instinto de conexión vertical
        self.tropism_strength = 0.3 

    def mutate_new_neuron(self, net, idx, parent_idx=None):
        """
        Refina el ADN y establece la firma química (Excitatoria vs Inhibitoria).
        v3.5: Codificación de identidad funcional.
        """
        # --- HERENCIA Y MUTACIÓN ---
        if parent_idx is not None:
            parent_dna = net.dna[parent_idx]
            new_dna = []
            for gene in parent_dna:
                if random.random() < self.mutation_rate:
                    mutation = random.uniform(-0.15, 0.15)
                    new_dna.append(np.clip(gene + mutation, -1.0, 1.0))
                else:
                    new_dna.append(gene)
            net.dna[idx] = new_dna
        else:
            # Si no hay padre, el ADN base ya fue generado en __init__
            pass
        
        dna = net.dna[idx]
        
        # --- EXPRESIÓN GENÉTICA (ADN -> QUÍMICA) ---
        gene_polaridad = dna[3]
        gene_metabolismo = dna[4]
        gene_z_bias = dna[5] if len(dna) > 5 else random.uniform(0.1, 1.0)
        
        # NT 0: Glutamato (Excitación)
        excitatory = max(0.05, gene_polaridad) if gene_polaridad > 0 else 0.05
        # NT 1: GABA (Inhibición)
        inhibitory = abs(gene_polaridad) if gene_polaridad < 0 else 0.05
        
        raw_nt = [
            excitatory,                      # NT 0: Excitación
            inhibitory,                      # NT 1: Inhibición Lateral
            abs(gene_metabolismo),           # NT 2: Sensibilidad a Dopamina
            abs(gene_polaridad * gene_metabolismo), # NT 3: Estabilidad Sináptica
            gene_z_bias                      # NT 4: Tropismo (Crecimiento)
        ]

        # Normalización del vector químico
        total = sum(raw_nt) or 1.0
        net.nt_vector[idx] = [val / total for val in raw_nt]

        # --- CONFIGURACIÓN DE RECEPTORES ---
        if inhibitory > excitatory:
            net.receptors[idx] = [0.8, 0.1, 0.4, 0.2, 0.5] # Fuerte respuesta al Glutamato
        else:
            net.receptors[idx] = [0.3, 0.6, 0.4, 0.2, 0.5] # Fuerte respuesta al GABA (Auto-regulación)
    
    def apply_evolution_step(self, net):
        """
        Evolución basada en el éxito de la Mirada.
        Las neuronas que ayudan a mover la mirada desde 0.50 sobreviven más.
        """
        fitness_scores = []
        
        # Calculamos el éxito de la red (distancia Mirada vs Foco)
        try:
            eye_x, eye_y = net.vision.current_eye_pos if hasattr(net, 'vision') else (0.5, 0.5)
            foci = net.vision.last_foci if hasattr(net, 'vision') else [(0.5, 0.5)]
            
            # Recompensa por reducir el error del 0.50
            global_reward = 0.1
            if foci:
                # 🔥 CORRECCIÓN: Normalización dinámica usando GRID_SIZE
                dist = np.sqrt((eye_x - foci[0][0]/GRID_SIZE)**2 + (eye_y - foci[0][1]/GRID_SIZE)**2)
                global_reward = max(0.1, 1.0 - dist)
        except:
            global_reward = 0.1

        for i in range(net.n):
            if net.active[i]:
                # FITNESS = Energía + Eficiencia + (Dopamina Regional * Recompensa Global)
                r_idx = net.neuron_region[i]
                reg_dopamine = net.regions[r_idx].dopamine if r_idx < len(net.regions) else 0
                
                score = (net.energy[i] * 0.5) + (reg_dopamine * global_reward * 0.5)
                fitness_scores.append((i, score))
        
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        top_count = max(1, len(fitness_scores) // 10)
        winners = [idx for idx, score in fitness_scores[:top_count]]

        if winners:
            self._horizontal_transfer(net, winners)

    def _horizontal_transfer(self, net, winners):
        """Transferencia de rasgos: Las neuronas 'ciegas' aprenden de las que 'ven'."""
        for _ in range(10):
            source = random.choice(winners)
            target = random.randint(0, net.n - 1)
            
            if net.active[target] and source != target:
                # El target absorbe el ADN del ganador
                for g in range(len(net.dna[target])):
                    if random.random() < 0.2:
                        net.dna[target][g] = (net.dna[target][g] * 0.8) + (net.dna[source][g] * 0.2)
                
                # --- AQUÍ APLICAMOS LA MUTACIÓN FÍSICA ---
                self._sync_neuron_physics(net, target)

    def _sync_neuron_physics(self, net, idx):
        """
        Sincroniza los parámetros físicos (tau_m, v_thresh, refr_period) 
        con el ADN actual de la neurona.
        """
        dna = net.dna[idx]
        polaridad = dna[3] # Usamos el gen 3 como selector
        
        # Seleccionamos perfil basado en el gen de polaridad
        if polaridad > 0.5:
            perfil = EVO_MENU["RAPID_FIRE"]
        elif polaridad < -0.5:
            perfil = EVO_MENU["BUFFER"]
        else:
            perfil = EVO_MENU["RELAY"]
            
        # Actualizamos los vectores de estado de la red
        net.tau_m[idx] = perfil.tau_m
        net.v_thresh[idx] = perfil.v_thresh
        net.refr_period[idx] = perfil.refr_period