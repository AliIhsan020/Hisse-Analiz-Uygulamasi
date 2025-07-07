# Gelişmiş 9 Renkli Boid Simülasyonu
# Yön tuşları ile algılama mesafesini değiştir, 1-9 ile renk seçimi, 0 ile tüm renkler
# Fare ile boidleri çekebilir veya itebilirsin
# Shift + mouse tekerleği ile boid sayısını değiştir, + ve - ile boid boyutunu ayarla
# R ile boidleri sıfırla, Space ile duraklat/başlat, T ile algılama alanını göster/gizle
# I ile iz bırakmayı aç/kapat, A ile yeni boid ekle, S ile boid sil
# D ile tehlike noktası, orta mouse ile yem noktası ekle
# Renk Hiyerarşisi: Siyah(1) -> Mor(2) -> Mavi(3) -> Yeşil(4) -> Sarı(5) -> Turuncu(6) -> Kırmızı(7) -> Pembe(8) -> Beyaz(9)
# Koyu renkler açık renklerden kaçar, boidler kendi renkleriyle gruplaşır

import pygame
import random
import math

pygame.init()

WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gelişmiş 9 Renkli Boid Simülasyonu")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# 9 Renk Sistemi: Siyahtan Beyaza Hiyerarşi
COLORS = {
    1: (0, 0, 0),        # Siyah
    2: (128, 0, 128),    # Mor
    3: (0, 0, 255),      # Mavi
    4: (0, 255, 0),      # Yeşil
    5: (255, 255, 0),    # Sarı
    6: (255, 165, 0),    # Turuncu
    7: (255, 0, 0),      # Kırmızı
    8: (255, 192, 203),  # Pembe
    9: (255, 255, 255),  # Beyaz
    0: (200, 200, 200)   # Tümü (gri)
}

color_names = {
    1: "SİYAH",
    2: "MOR",
    3: "MAVİ",
    4: "YEŞİL",
    5: "SARI",
    6: "TURUNCU",
    7: "KIRMIZI",
    8: "PEMBE",
    9: "BEYAZ",
    0: "TÜMÜ"
}

selected_color = 0

NUM_BOIDS = 60
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
        self.color_key = random.randint(1, 9)  # 1-9 arası renk
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
            if other != self:
                escape_needed = False
                
                # Renk hiyerarşisi kontrolü: Koyu renkler açık renklerden kaçar
                if self.color_key < other.color_key:  # Bu boid daha koyu
                    escape_needed = True
                elif self.color_key == other.color_key:  # Aynı renk - klasik separation
                    escape_needed = True

                if escape_needed:
                    distance = self.position.distance_to(other.position)
                    if distance < PERCEPTION_RADIUS:
                        diff = self.position - other.position
                        if distance > 0:
                            # Hiyerarşi farkı ne kadar büyükse kaçma kuvveti o kadar güçlü
                            hierarchy_multiplier = 1.0
                            if self.color_key < other.color_key:
                                hierarchy_multiplier = 1.0 + (other.color_key - self.color_key) * 0.3
                            diff = diff.normalize() * hierarchy_multiplier / distance
                        steering += diff
                        total += 1
        
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering = steering.normalize() * MAX_SPEED
                steering -= self.velocity
                steering *= 1.8  # Separation kuvvetini artır
                if steering.length() > MAX_FORCE * 2.5:
                    steering = steering.normalize() * MAX_FORCE * 2.5
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

        if attract_point and (selected_color == 0 or self.color_key == selected_color):
            self.acceleration += self.attracted_to(attract_point)
        if repel_point and (selected_color == 0 or self.color_key == selected_color):
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
            if len(self.trail) > 30:
                self.trail.pop(0)
        else:
            self.trail = []

    def draw(self, screen, boid_size=8):
        # İz çiz
        if show_trail and len(self.trail) > 1:
            for i in range(len(self.trail) - 1):
                start_pos = (int(self.trail[i].x), int(self.trail[i].y))
                end_pos = (int(self.trail[i + 1].x), int(self.trail[i + 1].y))
                # Teleport kontrolü
                if abs(start_pos[0] - end_pos[0]) < WIDTH/2 and abs(start_pos[1] - end_pos[1]) < HEIGHT/2:
                    alpha = int(255 * (i + 1) / len(self.trail))
                    trail_color = (*self.color[:3], alpha)
                    pygame.draw.line(screen, self.color, start_pos, end_pos, 2)

        # Boid gövdesi
        angle = math.degrees(math.atan2(-self.velocity.y, self.velocity.x))
        cx = self.position.x % WIDTH
        cy = self.position.y % HEIGHT
        
        # Üçgen şekli
        points = [
            (cx + math.cos(math.radians(angle)) * boid_size,
             cy - math.sin(math.radians(angle)) * boid_size),
            (cx + math.cos(math.radians(angle + 140)) * (boid_size * 0.75),
             cy - math.sin(math.radians(angle + 140)) * (boid_size * 0.75)),
            (cx + math.cos(math.radians(angle - 140)) * (boid_size * 0.75),
             cy - math.sin(math.radians(angle - 140)) * (boid_size * 0.75))
        ]
        
        # Siyah boidler için beyaz kenar çiz
        if self.color_key == 1:
            pygame.draw.polygon(screen, (255, 255, 255), points, 2)
        else:
            pygame.draw.polygon(screen, self.color, points)

boids = [Boid() for _ in range(NUM_BOIDS)]

def reset_boids():
    global boids
    boids = [Boid() for _ in range(NUM_BOIDS)]

def draw_color_legend():
    # Renk efsanesi
    legend_y = 50
    for i in range(1, 10):
        color = COLORS[i]
        name = color_names[i]
        
        # Renk kutusu
        rect = pygame.Rect(10, legend_y + (i-1) * 25, 20, 20)
        if i == 1:  # Siyah için beyaz kenar
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        else:
            pygame.draw.rect(screen, color, rect)
        
        # Renk adı
        text_color = (255, 255, 255) if i == 1 else color
        text = font.render(f"{i}: {name}", True, text_color)
        screen.blit(text, (35, legend_y + (i-1) * 25 + 2))

def count_boids_by_color():
    counts = {}
    for i in range(1, 10):
        counts[i] = 0
    for boid in boids:
        counts[boid.color_key] += 1
    return counts

running = True
while running:
    screen.fill((30, 30, 30))
    keys = pygame.key.get_pressed()

    if not paused:
        # Algılama mesafesi kontrolü
        if keys[pygame.K_UP]:
            PERCEPTION_RADIUS += 1
        if keys[pygame.K_DOWN]:
            PERCEPTION_RADIUS -= 1
        PERCEPTION_RADIUS = max(10, min(200, PERCEPTION_RADIUS))

        # Boid boyutu kontrolü
        if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            boid_size = min(boid_size + 1, 20)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            boid_size = max(boid_size - 1, 3)

        # Fare etkileşimi
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
                        NUM_BOIDS = min(NUM_BOIDS + 5, 300)
                    else:
                        NUM_BOIDS = max(NUM_BOIDS - 5, 10)
                    reset_boids()
                else:
                    PERCEPTION_RADIUS += event.y * 5
                    PERCEPTION_RADIUS = max(10, min(200, PERCEPTION_RADIUS))
            elif event.type == pygame.KEYDOWN:
                # Renk seçimi (0-9)
                if event.key == pygame.K_0:
                    selected_color = 0
                elif event.key == pygame.K_1:
                    selected_color = 1
                elif event.key == pygame.K_2:
                    selected_color = 2
                elif event.key == pygame.K_3:
                    selected_color = 3
                elif event.key == pygame.K_4:
                    selected_color = 4
                elif event.key == pygame.K_5:
                    selected_color = 5
                elif event.key == pygame.K_6:
                    selected_color = 6
                elif event.key == pygame.K_7:
                    selected_color = 7
                elif event.key == pygame.K_8:
                    selected_color = 8
                elif event.key == pygame.K_9:
                    selected_color = 9
                elif event.key == pygame.K_r:
                    reset_boids()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_t:
                    show_perception = not show_perception
                elif event.key == pygame.K_i:
                    show_trail = not show_trail
                elif event.key == pygame.K_a:
                    NUM_BOIDS = min(NUM_BOIDS + 1, 300)
                    boids.append(Boid())
                elif event.key == pygame.K_s:
                    if len(boids) > 10:
                        NUM_BOIDS -= 1
                        boids.pop()
                elif event.key == pygame.K_d:
                    danger_point = pygame.Vector2(pygame.mouse.get_pos())
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:  # Orta teker: yem noktası
                    bait_point = pygame.Vector2(pygame.mouse.get_pos())

        # Boidleri güncelle ve çiz
        for boid in boids:
            if random.random() < 0.001:
                boid.velocity += pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
            boid.update(boids, attract_point, repel_point, bait_point, danger_point)
            boid.draw(screen, boid_size)
            
            # Algılama alanını göster
            if show_perception:
                pygame.draw.circle(screen, (80, 80, 80), (int(boid.position.x), int(boid.position.y)), PERCEPTION_RADIUS, 1)

        # Yem ve tehlike noktalarını çiz
        if bait_point is not None:
            pygame.draw.circle(screen, (0, 255, 0), (int(bait_point.x), int(bait_point.y)), 12)
            pygame.draw.circle(screen, (255, 255, 255), (int(bait_point.x), int(bait_point.y)), 12, 2)
        if danger_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), (int(danger_point.x), int(danger_point.y)), 12)
            pygame.draw.circle(screen, (255, 255, 255), (int(danger_point.x), int(danger_point.y)), 12, 2)

    else:
        # Duraklatıldığında
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused

        # Durgun boidleri çiz
        for boid in boids:
            boid.draw(screen, boid_size)
            if show_perception:
                pygame.draw.circle(screen, (80, 80, 80), (int(boid.position.x), int(boid.position.y)), PERCEPTION_RADIUS, 1)
        
        if bait_point is not None:
            pygame.draw.circle(screen, (0, 255, 0), (int(bait_point.x), int(bait_point.y)), 12)
        if danger_point is not None:
            pygame.draw.circle(screen, (255, 0, 0), (int(danger_point.x), int(danger_point.y)), 12)

    # Renk efsanesi çiz
    draw_color_legend()

    # Boid sayılarını göster
    counts = count_boids_by_color()
    count_text = "Boid Sayıları: "
    for i in range(1, 10):
        if counts[i] > 0:
            count_text += f"{color_names[i]}: {counts[i]} | "
    count_surface = font.render(count_text, True, (255, 255, 255))
    screen.blit(count_surface, (300, 50))

    # Ana bilgi paneli
    fps = int(clock.get_fps())
    display_color = COLORS[selected_color] if selected_color != 0 else (255, 255, 255)
    info = f"Seçili: {color_names[selected_color]} | Mesafe: {PERCEPTION_RADIUS} | Toplam: {len(boids)} | Boyut: {boid_size} | FPS: {fps}"
    info += f" | İz: {'Açık' if show_trail else 'Kapalı'} | {'DURAKLATILDI' if paused else 'ÇALIŞIYOR'}"
    text_surface = font.render(info, True, display_color)
    screen.blit(text_surface, (10, 10))

    # Kontrol bilgileri
    controls = "Kontroller: 0-9 (Renk) | Yön tuşları (Mesafe) | +/- (Boyut) | R (Sıfırla) | Space (Durdur) | T (Algılama) | I (İz)"
    control_surface = font.render(controls, True, (180, 180, 180))
    screen.blit(control_surface, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()