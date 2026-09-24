# core/module.py
# Sistema de puertos y módulos plug-and-play con soporte para coordenadas 3D

import random

class Port:
    def __init__(self, name, module, is_input=True, strength=1.0, pos=None):
        self.name = name
        self.module = module
        self.is_input = is_input
        self.strength = max(0.1, min(2.0, strength))  # límites razonables
        self.value = 0.0
        self.connected_neurons = []  # lista de índices de neuronas
        
        # --- NUEVO: Posición 3D para el visualizador ---
        # Si no se define, se coloca en el centro por defecto (25, 25, 25)
        self.pos = pos if pos is not None else (25.0, 25.0, 25.0)
    
    def to_dict(self):
        """Estado serializable del puerto — sin locks, sin referencias circulares."""
        return {
            "strength": self.strength,
            "value": float(self.value),
            "connected_neurons": [int(i) for i in self.connected_neurons],
        }

    def load_dict(self, data, max_idx=None):
        self.strength = data.get("strength", self.strength)
        self.value = data.get("value", self.value)
        neuronas = data.get("connected_neurons", [])
        if max_idx is not None:
            neuronas = [i for i in neuronas if 0 <= i < max_idx]
        self.connected_neurons = list(neuronas)

    def get_all_ports(self):
        """Retorna una lista con todos los puertos (In y Out)"""
        return list(self.input_ports.values()) + list(self.output_ports.values())
    
    def connect(self, neuron_idx):
        if neuron_idx not in self.connected_neurons:
            self.connected_neurons.append(neuron_idx)

    def disconnect(self, neuron_idx):
        if neuron_idx in self.connected_neurons:
            self.connected_neurons.remove(neuron_idx)

    def mutate(self):
        """Mutación de la fuerza del puerto (plasticidad estructural)"""
        if random.random() < 0.08:  # ~8% chance por ciclo de mutación
            self.strength += random.uniform(-0.12, 0.12)
            self.strength = max(0.1, min(2.0, self.strength))


class Module:
    def __init__(self, name):
        self.name = name
        self.input_ports = {}
        self.output_ports = {}
        self.active = True

    
    def get_state(self):
        """Estado serializable de TODOS los puertos (in + out). Genérico: cualquier
        módulo nuevo lo hereda gratis, sin escribir lógica de guardado propia."""
        return {
            "input_ports":  {name: p.to_dict() for name, p in self.input_ports.items()},
            "output_ports": {name: p.to_dict() for name, p in self.output_ports.items()},
        }

    def restore_state(self, data, max_idx=None):
        """Aplica un estado guardado sobre los puertos YA CREADOS del módulo actual.
        Si un puerto del guardado ya no existe, se ignora. Si hay uno nuevo, 
        queda con su valor por defecto."""
        for name, pdata in data.get("input_ports", {}).items():
            if name in self.input_ports:
                self.input_ports[name].load_dict(pdata, max_idx=max_idx)
                
        for name, pdata in data.get("output_ports", {}).items():
            if name in self.output_ports:
                self.output_ports[name].load_dict(pdata, max_idx=max_idx)


    def add_input_port(self, name, strength=1.0, pos=None):
        """Añade un puerto de entrada con posición opcional para el visualizador"""
        port = Port(name, self, is_input=True, strength=strength, pos=pos)
        self.input_ports[name] = port
        return port

    def add_output_port(self, name, strength=1.0, pos=None):
        """Añade un puerto de salida con posición opcional para el visualizador"""
        port = Port(name, self, is_input=False, strength=strength, pos=pos)
        self.output_ports[name] = port
        return port

    def update(self, net, t):
        """Cada módulo específico (como HeartModule) implementa su lógica aquí"""
        raise NotImplementedError("Debes implementar el método update en la subclase.")

    def mutate(self):
        """Mutación genérica de todos los puertos del módulo"""
        for port in list(self.input_ports.values()) + list(self.output_ports.values()):
            port.mutate()