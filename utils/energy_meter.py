"""
Theoretical energy accounting for spiking neural networks.

Energy model follows the standard neuromorphic cost split:
    - AC (accumulate) per active synaptic operation (spike)
    - MAC (multiply-accumulate) per dense floating-point operation
"""


class EnergyMeter:
    def __init__(self, e_mac_pj: float = 4.6, e_ac_pj: float = 0.9):
        self.e_mac_pj = e_mac_pj
        self.e_ac_pj = e_ac_pj

    def calculate_sops(self, spikes, fan_out: int) -> int:
        """Total active synaptic operations = number of spikes * fan-out."""
        return int(spikes.sum().item()) * fan_out

    def estimate_energy_pj(self, sops: int, flops: int) -> float:
        """Theoretical energy in picojoules."""
        return sops * self.e_ac_pj + flops * self.e_mac_pj
