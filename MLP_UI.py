from MultiLayerPerceptron import  MLP
import pygame

Digit_NN = MLP([28 * 28, 128, 64, 10])

Digit_NN.load_parameters_text('DigitRecogNets/digit_recog_9.txt')

pygame.init()
scale = 20
screen = pygame.display.set_mode((28*scale + 200,28*scale))
color_locations = []

font = pygame.font.SysFont("Helvetica",30)

rect = pygame.Rect(0, 0, 28*scale, 28*scale)
sub = screen.subsurface(rect)

def get_prediction():
    Digit_NN.input_from_image('digit_image.jpg',(28,28))
    return Digit_NN.get_output()

def display_prediction(prediction):
    white = (255,255,255)
    grey = (125,125,125)
    density = 7
    pygame.draw.line(screen,white,(28*scale,0),(28*scale,28*scale))
    for i in range(1,density):
        pygame.draw.line(screen,grey,(28*scale,i*28*scale//density),(0,i*28*scale//density))
        pygame.draw.line(screen,grey,(i*28*scale//density,0),(i*28*scale//density,28*scale))
    y_offset = 80
    spacing = 40
    x_offset = 28*scale + 30
    pred = Digit_NN.get_prediction()
    for i in range(10):
        if i == pred:
            color = (5, 173, 56)
        else:
            color = white
        screen.blit(font.render(f"{i} : {round(100*prediction[i],2)}%",True,color),(x_offset,y_offset+spacing*i))

clock = pygame.time.Clock()
running = True
frame = 0
while running:
    clock.tick(120)
    frame += 1
    screen.fill((0,0,0))

    for position in color_locations:
        pygame.draw.circle(screen,(255,255,255),position,scale)

    if frame%20 == 0:
        pygame.image.save(sub,'digit_image.jpg')
        Digit_NN.input_from_image('digit_image.jpg',(28,28))
        frame = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                color_locations = []

    if pygame.mouse.get_pressed(num_buttons=3)[0]:
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos not in color_locations:
            color_locations.append(mouse_pos)
    pygame.draw.rect(screen,(0,0,0),pygame.Rect((scale*28,0,200,scale*28)))

    display_prediction(Digit_NN.get_output())
    pygame.display.update()

pygame.quit()
