# Basit Boid (Parçacık) Simülasyonu
# Yön tuşlarıyla algılama mesafesini değiştir, 1-2-3-4 ile renk seç, fare ile çek/it
# Shift + mouse tekerleği ile boid sayısını değiştir, + ve - ile boid boyutunu ayarla
# R ile boidleri sıfırla, Space ile duraklat/başlat, T ile algılama alanını göster/gizle
# A ile yeni boid ekle, S ile boid sil, boidler kendi renkleriyle gruplaşır ve ayrılır
# FPS, boid sayısı, seçili renk ve diğer bilgiler ekranda gösterilir

import pygame
import random
import math

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Basit Boid Simülasyonu")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

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
selected_color = 4

NUM_BOIDS = 50
MAX_SPEED = 4
MAX_FORCE = 0.05
PERCEPTION_RADIUS = 50

boid_size = 8
show_perception = False
paused = False
show_trail = False

# Yem ve tehlike noktaları
bait_point = None
danger_point = None

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, MAX_SPEED)
        self.acceleration = pygame.Vector2(0, 0)
        self.color_key = random.randint(1, 3)
        self.color = COLORS[self.color_key]
        self.trail = []

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
            # Gökkuşağı sırası: Daha sağdaki renkler soldakilerden kaçar
            if other != self:
                rainbow_escape = False
                if self.color_key == 3 and other.color_key in [1, 2]:  # Mavi, kırmızı ve yeşilden kaçar
                    rainbow_escape = True
                elif self.color_key == 2 and other.color_key == 1:     # Yeşil, kırmızıdan kaçar
                    rainbow_escape = True
                elif self.color_key == other.color_key:                # Aynı renkler klasik separation
                    rainbow_escape = True

                if rainbow_escape:
                    distance = self.position.distance_to(other.position)
                    if distance < PERCEPTION_RADIUS:
                        diff = self.position - other.position
                        if distance > 0:
                            diff /= (distance * distance)
                        steering += diff
                        total += 1
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering = steering.normalize() * MAX_SPEED
                steering -= self.velocity
                steering *= 1.5
                if steering.length() > MAX_FORCE * 2:
                    steering = steering.normalize() * MAX_FORCE * 2
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

    def update(self, boids, attract_point=None, repel_point=None, bait=None, danger=None):
        self.acceleration = pygame.Vector2(0, 0)
        self.acceleration += self.align(boids)
        self.acceleration += self.cohesion(boids)
        self.acceleration += self.separation(boids)

        if attract_point and (selected_color == 4 or self.color_key == selected_color):
            self.acceleration += self.attracted_to(attract_point)
        if repel_point and (selected_color == 4 or self.color_key == selected_color):
            self.acceleration += self.repelled_from(repel_point)

        # Yem noktası (her boid yaklaşır)
        if bait is not None:
            self.acceleration += self.attracted_to(bait, strength=0.3)
        # Tehlike noktası (her boid kaçar)
        if danger is not None:
            self.acceleration += self.repelled_from(danger, strength=0.5)

        self.velocity += self.acceleration
        if self.velocity.length() > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED
        self.position += self.velocity
        self.edges()

        # İz bırakma
        if show_trail:
            self.trail.append(self.position.copy())
            if len(self.trail) > 40:
                self.trail.pop(0)
        else:
            self.trail = []

    def draw(self, screen, boid_size=8):
        global show_trail
        # İz çiz
        if show_trail and len(self.trail) > 1:
            last_point = None
            segment = []
            for p in self.trail:
                x = p.x % WIDTH
                y = p.y % HEIGHT
                current_point = (int(x), int(y))
                if last_point is not None:
                    dx = abs(p.x - last_point[0])
                    dy = abs(p.y - last_point[1])
                    # Teleport (wrap) kontrolü
                    if dx > WIDTH / 2 or dy > HEIGHT / 2:
                        # Teleport öncesi: iz bırakmayı kapat
                        prev_show_trail = show_trail
                        show_trail = False
                        if len(segment) > 1:
                            pygame.draw.lines(screen, self.color, False, segment, 2)
                        segment = []
                        # Teleport sonrası: iz bırakmayı tekrar aç
                        show_trail = prev_show_trail
                segment.append(current_point)
                last_point = (p.x, p.y)
            if len(segment) > 1:
                pygame.draw.lines(screen, self.color, False, segment, 2)
        # Boid gövdesi
        angle = math.degrees(math.atan2(-self.velocity.y, self.velocity.x))
        cx = self.position.x % WIDTH
        cy = self.position.y % HEIGHT
        points = [
            (cx + math.cos(math.radians(angle)) * boid_size,
             cy - math.sin(math.radians(angle)) * boid_size),
            (cx + math.cos(math.radians(angle + 140)) * (boid_size * 0.75),
             cy - math.sin(math.radians(angle + 140)) * (boid_size * 0.75)),
            (cx + math.cos(math.radians(angle - 140)) * (boid_size * 0.75),
             cy - math.sin(math.radians(angle - 140)) * (boid_size * 0.75))
        ]
        pygame.draw.polygon(screen, self.color, points)

boids = [Boid() for _ in range(NUM_BOIDS)]

def reset_boids():
    global boids
    boids = [Boid() for _ in range(NUM_BOIDS)]

running = True
while running:
    screen.fill((30, 30, 30))
    keys = pygame.key.get_pressed()

    if not paused:
        if keys[pygame.K_UP]:
            PERCEPTION_RADIUS += 1
        if keys[pygame.K_DOWN]:
            PERCEPTION_RADIUS -= 1
        PERCEPTION_RADIUS = max(10, min(200, PERCEPTION_RADIUS))

        if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            boid_size = min(boid_size + 1, 20)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            boid_size = max(boid_size - 1, 3)

        attract_point = None
        repel_point = None
        mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
        if pygame.mouse.get_pressed()[0]:
            attract_point = mouse_pos
        if pygame.mouse.get_pressed()[2]:
            repel_point = mouse_pos

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    if event.y > 0:
                        NUM_BOIDS = min(NUM_BOIDS + 5, 200)
                    else:
                        NUM_BOIDS = max(NUM_BOIDS - 5, 5)
                    reset_boids()
                else:
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
                elif event.key == pygame.K_r:
                    reset_boids()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    show_perception = not show_perception
                elif event.key == pygame.K_i:
                    show_trail = not show_trail
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                    boid_size = min(boid_size + 1, 20)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    boid_size = max(boid_size - 1, 3)
                elif event.key == pygame.K_a:
                    NUM_BOIDS = min(NUM_BOIDS + 1, 200)
                    boids.append(Boid())
                elif event.key == pygame.K_s:
                    if len(boids) > 5:
                        NUM_BOIDS -= 1
                        boids.pop()
                elif event.key == pygame.K_d:
                    # Tehlike noktası ekle (fare konumuna)
                    danger_point = pygame.Vector2(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # Orta teker: yem noktası ekle
                    bait_point = pygame.Vector2(pygame.mouse.get_pos())

        # Güncelle ve çiz
        for boid in boids:
            if random.random() < 0.002:
                boid.velocity += pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            boid.update(boids, attract_point, repel_point, bait_point, danger_point)
            boid.draw(screen, boid_size)
            if show_perception:
                pygame.draw.circle(screen, (100, 100, 100), (int(boid.position.x), int(boid.position.y)), PERCEPTION_RADIUS, 1)

        # Yem ve tehlike noktalarını çiz
        if bait_point is not None:
            pygame.draw.circle(screen, (0, 255, 0), (int(bait_point.x), int(bait_point.y)), 10)
        if danger_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), (int(danger_point.x), int(danger_point.y)), 10)

    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        for boid in boids:
            boid.draw(screen, boid_size)
            if show_perception:
                pygame.draw.circle(screen, (100, 100, 100), (int(boid.position.x), int(boid.position.y)), PERCEPTION_RADIUS, 1)
        if bait_point is not None:
            pygame.draw.circle(screen, (0, 255, 0), (int(bait_point.x), int(bait_point.y)), 10)
        if danger_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), (int(danger_point.x), int(danger_point.y)), 10)

    # FPS ve bilgiler
    fps = int(clock.get_fps())
    info = f"Seçili Renk: {color_names[selected_color]} | Mesafe: {PERCEPTION_RADIUS} | Boid: {len(boids)} | Boyut: {boid_size} | FPS: {fps} | İz: {'Açık' if show_trail else 'Kapalı'}"
    text_surface = font.render(info, True, COLORS[selected_color] if selected_color != 4 else (255, 255, 255))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()