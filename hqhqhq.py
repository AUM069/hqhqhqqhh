import streamlit as st
import pygame
import random
import sys
from PIL import Image
import multiprocessing
import os

# Pygame game function
def run_game():
    # Initialize Pygame
    pygame.init()

    # Screen dimensions
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("💖 Shreya's Valentine's Temple Run 💖")

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    # Load images (hardcoded paths)
    player_image = pygame.image.load("/Users/aumwaghmare/Downloads/player.jpeg")  # Replace with your photo
    obstacle_image = pygame.image.load("/Users/aumwaghmare/Downloads/object.jpeg")  # Replace with an obstacle image
    background_image = pygame.image.load("/Users/aumwaghmare/Downloads/bg2.jpeg")  # Replace with a romantic background
    valentine_image = pygame.image.load("/Users/aumwaghmare/Downloads/spiderman.jpeg")  # Replace with a photo for the ending
    heart_image = pygame.image.load("/Users/aumwaghmare/Downloads/lesgogoggo.png")  # Add a heart image (you can download one)

    # Resize images
    player_image = pygame.transform.scale(player_image, (50, 50))
    obstacle_image = pygame.transform.scale(obstacle_image, (50, 50))
    background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))
    valentine_image = pygame.transform.scale(valentine_image, (WIDTH, HEIGHT))
    heart_image = pygame.transform.scale(heart_image, (30, 30))

    # Player settings
    player_x = 100
    player_y = HEIGHT // 2
    player_speed = 5
    player_size = 50  # Initial size of the player

    # Obstacle settings
    obstacle_width = 50
    obstacle_height = 50
    obstacle_x = WIDTH
    obstacle_y = random.randint(0, HEIGHT - obstacle_height)
    obstacle_speed = 7

    # Heart settings
    heart_width = 30
    heart_height = 30
    heart_x = random.randint(0, WIDTH - heart_width)
    heart_y = random.randint(0, HEIGHT - heart_height)
    hearts_collected = 0

    # Game variables
    score = 0
    font = pygame.font.SysFont("Arial", 35)

    # Function to display text on the screen
    def draw_text(screen, text, font, color, x, y):
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, (x, y))

    # Game loop
    clock = pygame.time.Clock()
    running = True
    game_over = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            # Move player
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and player_y > 0:
                player_y -= player_speed
            if keys[pygame.K_DOWN] and player_y < HEIGHT - player_size:
                player_y += player_speed

            # Move obstacle
            obstacle_x -= obstacle_speed
            if obstacle_x < 0:
                obstacle_x = WIDTH
                obstacle_y = random.randint(0, HEIGHT - obstacle_height)
                score += 1

            # Move heart
            heart_x -= 5  # Hearts move slower than obstacles
            if heart_x < 0:
                heart_x = WIDTH
                heart_y = random.randint(0, HEIGHT - heart_height)

            # Collision detection with obstacle
            player_rect = pygame.Rect(player_x, player_y, player_size, player_size)
            obstacle_rect = pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)
            if player_rect.colliderect(obstacle_rect):
                game_over = True

            # Collision detection with heart
            heart_rect = pygame.Rect(heart_x, heart_y, heart_width, heart_height)
            if player_rect.colliderect(heart_rect):
                hearts_collected += 1
                player_size += 5  # Increase player size
                player_image = pygame.transform.scale(pygame.image.load("player.png"), (player_size, player_size))
                heart_x = WIDTH  # Move heart off-screen to respawn
                heart_y = random.randint(0, HEIGHT - heart_height)

            # Draw everything
            screen.blit(background_image, (0, 0))
            screen.blit(player_image, (player_x, player_y))
            screen.blit(obstacle_image, (obstacle_x, obstacle_y))
            screen.blit(heart_image, (heart_x, heart_y))
            draw_text(screen, f"Score: {score}", font, WHITE, 10, 10)
            draw_text(screen, f"Hearts: {hearts_collected}", font, RED, 10, 50)

        else:
            # Game over screen
            screen.blit(valentine_image, (0, 0))
            draw_text(screen, "Will you be my Valentine jiiii ???/", font, RED, WIDTH // 4, HEIGHT // 2)
            draw_text(screen, f"Final Score: {score}", font, RED, WIDTH // 4, HEIGHT // 2 + 50)
            draw_text(screen, f"catch kar liya jii : {hearts_collected}", font, RED, WIDTH // 4, HEIGHT // 2 + 100)

        pygame.display.update()
        clock.tick(30)

    pygame.quit()
    sys.exit()

# Streamlit interface
def main():
    st.title("💖 Shreya's game  💖")
    st.markdown("## Shreyaaaa Tiwari, are you ready to play?")
    st.markdown("### Are you **fucking** ready? 😈")

    if st.button("Start Game"):
        # Run the Pygame game in a separate process
        process = multiprocessing.Process(target=run_game)
        process.start()
        process.join()  # Wait for the game to finish

if __name__ == "__main__":
    main()
