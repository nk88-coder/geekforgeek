import pygame
import math
import sys  # for exit

try:
    # Initialize
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Bike Game")

    # Load bike image
    bike_img = pygame.image.load(r"C:\Users\Nishok\Desktop\KAWASAKIgame.jpg")
    bike_img = pygame.transform.scale(bike_img, (100, 50))  # scale to fit screen nicely

    # Bike variables
    x, y = WIDTH//2, HEIGHT//2
    angle = 0
    speed = 0
    MAX_SPEED = 10
    ACCEL = 0.2
    FRICTION = 0.05

    clock = pygame.time.Clock()
    run = True

    while run:
        dt = clock.tick(60) / 1000  # delta time
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            speed += ACCEL
        elif keys[pygame.K_DOWN]:
            speed -= ACCEL
        else:
            speed *= (1 - FRICTION)

        if keys[pygame.K_LEFT]:
            angle += 3  # degrees
        if keys[pygame.K_RIGHT]:
            angle -= 3

        # Move bike
        rad = math.radians(angle)
        x += speed * math.cos(rad)
        y -= speed * math.sin(rad)

        # Draw
        win.fill((50, 150, 50))  # green background
        rotated_bike = pygame.transform.rotate(bike_img, angle)
        rect = rotated_bike.get_rect(center=(x, y))
        win.blit(rotated_bike, rect)

        pygame.display.update()

except Exception as e:
    print("Error:", e)
    input("Press Enter to exit...")
    sys.exit()
finally:
    pygame.quit()

