import pygame
import sys

# Oyun ayarları
WIDTH, HEIGHT = 600, 600
FPS = 60
PLAYER_SIZE = 40
PLAYER_SPEED = 5

# Renkler
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pacman Tarzı Oyun")
clock = pygame.time.Clock()

# Oyuncu başlangıç konumu
player_x = WIDTH // 2
player_y = HEIGHT // 2

# Ana döngü
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Tuş kontrolleri
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x -= PLAYER_SPEED
    if keys[pygame.K_RIGHT]:
        player_x += PLAYER_SPEED
    if keys[pygame.K_UP]:
        player_y -= PLAYER_SPEED
    if keys[pygame.K_DOWN]:
        player_y += PLAYER_SPEED

    # Kenarlardan çıkmayı engelle
    player_x = max(0, min(WIDTH - PLAYER_SIZE, player_x))
    player_y = max(0, min(HEIGHT - PLAYER_SIZE, player_y))

    # Ekranı temizle
    screen.fill(BLACK)

    # Oyuncuyu çiz
    pygame.draw.circle(screen, YELLOW, (player_x + PLAYER_SIZE // 2, player_y + PLAYER_SIZE // 2), PLAYER_SIZE // 2)

    pygame.display.flip()
    clock.tick(FPS)