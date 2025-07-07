import pygame
import random
import math

pygame.init()

# Pygame ayarları
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Simülasyonu (Skilli)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# Renkler
COLORS = {
    1: (255, 0, 0),    # Kırmızı
    2: (0, 255, 0),    # Yeşil
    3: (0, 150, 255),  # Mavi
    4: (255, 255, 255) # Tümü
}
color_names = {
    1: "KIRMIZI",
    2: "YEŞİL",
    3: "MAVİ",
    4: "TÜMÜ"
}
selected_color = 4  # Başlangıçta tümü seçili

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
        self.color_key = random.randint(1, 3)
        self.color = COLORS[self.color_key]

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
            if other.color_key == self.color_key:
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
            if other.color_key == self.color_key:
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
            if other.color_key == self.color_key:
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

    def attracted_to(self, point, strength=0.2):
        force = point - self.position
        distance = force.length()
        if distance > 0:
            force = force.normalize() * strength
            return force
        return pygame.Vector2(0, 0)

    def repelled_from(self, point, strength=0.3):
        force = self.position - point
        distance = force.length()
        if distance < 150 and distance > 0:
            force = force.normalize() * (strength * (150 - distance) / 150)
            return force
        return pygame.Vector2(0, 0)

    def update(self, boids, attract_point=None, repel_point=None):
        self.acceleration = pygame.Vector2(0, 0)
        self.acceleration += self.align(boids)
        self.acceleration += self.cohesion(boids)
        self.acceleration += self.separation(boids)

        if attract_point and (selected_color == 4 or self.color_key == selected_color):
            self.acceleration += self.attracted_to(attract_point)
        if repel_point and (selected_color == 4 or self.color_key == selected_color):
            self.acceleration += self.repelled_from(repel_point)

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
    screen.fill((30, 30, 30))
    keys = pygame.key.get_pressed()

    # Yön tuşları ve scroll ile PERCEPTION_RADIUS ayarı
    if keys[pygame.K_UP]:
        PERCEPTION_RADIUS += 1
    if keys[pygame.K_DOWN]:
        PERCEPTION_RADIUS -= 1
    PERCEPTION_RADIUS = max(10, min(200, PERCEPTION_RADIUS))

    # Mouse olayları
    attract_point = None
    repel_point = None
    mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
    if pygame.mouse.get_pressed()[0]:  # Sol tık
        attract_point = mouse_pos
    if pygame.mouse.get_pressed()[2]:  # Sağ tık
        repel_point = mouse_pos

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            PERCEPTION_RADIUS += event.y * 5
            PERCEPTION_RADIUS = max(10, min(200, PERCEPTION_RADIUS))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                selected_color = 1
            elif event.key == pygame.K_2:
                selected_color = 2
            elif event.key == pygame.K_3:
                selected_color = 3
            elif event.key == pygame.K_4:
                selected_color = 4

    # Güncelle ve çiz
    for boid in boids:
        boid.update(boids, attract_point, repel_point)
        boid.draw(screen)

    # Seçili rengi ekrana yaz
    text_surface = font.render(f"Seçili Renk: {color_names[selected_color]} | Mesafe: {PERCEPTION_RADIUS}", True, COLORS[selected_color] if selected_color != 4 else (255, 255, 255))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
