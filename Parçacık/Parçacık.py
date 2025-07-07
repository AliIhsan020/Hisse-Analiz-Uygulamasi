import pygame
import random
import math

# Pygame ayarları
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Simülasyonu")
clock = pygame.time.Clock()

# Renkler (3 tür)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 150, 255)]

# Boid ayarları
NUM_BOIDS = 50
MAX_SPEED = 4
MAX_FORCE = 0.05
PERCEPTION_RADIUS = 50

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, MAX_SPEED)
        self.acceleration = pygame.Vector2(0, 0)
        self.color = random.choice(COLORS)

    def edges(self):
        if self.position.x > WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = WIDTH
        if self.position.y > HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = HEIGHT

    def align(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        avg_vector = pygame.Vector2(0, 0)
        for other in boids:
            if other.color == self.color:  # sadece aynı renkteki boids
                distance = self.position.distance_to(other.position)
                if other != self and distance < PERCEPTION_RADIUS:
                    avg_vector += other.velocity
                    total += 1
        if total > 0:
            avg_vector /= total
            avg_vector = avg_vector.normalize() * MAX_SPEED
            steering = avg_vector - self.velocity
            if steering.length() > MAX_FORCE:
                steering = steering.normalize() * MAX_FORCE
        return steering

    def cohesion(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        center_of_mass = pygame.Vector2(0, 0)
        for other in boids:
            if other.color == self.color:
                distance = self.position.distance_to(other.position)
                if other != self and distance < PERCEPTION_RADIUS:
                    center_of_mass += other.position
                    total += 1
        if total > 0:
            center_of_mass /= total
            desired = center_of_mass - self.position
            if desired.length() > 0:
                desired = desired.normalize() * MAX_SPEED
                steering = desired - self.velocity
                if steering.length() > MAX_FORCE:
                    steering = steering.normalize() * MAX_FORCE
        return steering

    def separation(self, boids):
        steering = pygame.Vector2(0, 0)
        total = 0
        for other in boids:
            if other.color == self.color:
                distance = self.position.distance_to(other.position)
                if other != self and distance < PERCEPTION_RADIUS / 2:
                    diff = self.position - other.position
                    if distance > 0:
                        diff /= distance
                    steering += diff
                    total += 1
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering = steering.normalize() * MAX_SPEED
                steering -= self.velocity
                if steering.length() > MAX_FORCE:
                    steering = steering.normalize() * MAX_FORCE
        return steering

    def update(self, boids):
        self.acceleration = pygame.Vector2(0, 0)
        self.acceleration += self.align(boids)
        self.acceleration += self.cohesion(boids)
        self.acceleration += self.separation(boids)

        self.velocity += self.acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED
        self.position += self.velocity
        self.edges()

    def draw(self, screen):
        angle = math.degrees(math.atan2(-self.velocity.y, self.velocity.x))
        points = [
            (self.position.x + math.cos(math.radians(angle)) * 8,
             self.position.y - math.sin(math.radians(angle)) * 8),
            (self.position.x + math.cos(math.radians(angle + 140)) * 6,
             self.position.y - math.sin(math.radians(angle + 140)) * 6),
            (self.position.x + math.cos(math.radians(angle - 140)) * 6,
             self.position.y - math.sin(math.radians(angle - 140)) * 6)
        ]
        pygame.draw.polygon(screen, self.color, points)

# Boid listesi
boids = [Boid() for _ in range(NUM_BOIDS)]

# Ana döngü
running = True
while running:
    screen.fill((30, 30, 30))  # koyu arka plan
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for boid in boids:
        boid.update(boids)
        boid.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
