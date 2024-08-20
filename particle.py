import numpy as np

class Particle:
    mass = 0.0
    pos = np.ndarray((3,1))
	charge = 0.0

    def mass_set(self,m):
        self.mass = m
    def pos_set(self,p):
        self.pos = p
	def charge_set(self,c):
		self.charge = c



