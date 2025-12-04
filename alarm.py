import pygame
import sys

# تهيئة pygame
pygame.init()

# إعداد النافذة
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption("Alarm Player")

# تحميل الصوت (استعمال الملف الذي رفعته)
alarm_sound = r"/mnt/data/alarm.wav"   # ← لا تغيّر هذا

# تشغيل الصوت عند البداية
pygame.mixer.music.load(alarm_sound)
pygame.mixer.music.play(-1)  # تشغيل مستمر

running = True

while running:
    screen.fill((200, 200, 200))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # زر الخروج — تم تغييره إلى Q
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    pygame.display.flip()

pygame.mixer.music.stop()
pygame.quit()
sys.exit()
