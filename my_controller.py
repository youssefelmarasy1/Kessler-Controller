# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Aiman Saif, Youssef Elmarasy, Austin Newsham

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
import random
#import EasyGA
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
#from profcode import ScottDickController





class MyController(KesslerController):
 
  
    def __init__(self):
        self.eval_frames = 0 #What is this? The number of times the code iterated (Maximum is 60*30 = 1800)
        def generate_turn(low_limit, high_limit):
            NL_high = random.uniform(low_limit, random.uniform(-270, -180))
            NL_mid = random.uniform(low_limit, NL_high)
            NS_mid = random.uniform(NL_high, 0)
            PS_high = random.uniform(0, random.uniform(180, 270))
            PS_mid = random.uniform(0, PS_high)
            PL_mid = random.uniform(PS_high, high_limit)
            
            turn_list = [
                [low_limit, NL_mid, NL_high],
                [NL_high, NS_mid, 0],
                [0, PS_mid, PS_high],
                [PS_high, PL_mid, high_limit]
            ]
            
            return turn_list
        
        turn_list = generate_turn(-360, 360)
        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python
        #ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_turn = ctrl.Consequent(np.arange(-360,360,0.1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        #Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/3,-1*math.pi/6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/3,-1*math.pi/6,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/6,0,math.pi/6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,math.pi/6,math.pi/3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,math.pi/6,math.pi/3)
        
        #Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-90,-30])
        # ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90,-30,0])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-15,0,15])
        # ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0,30,90])
        # ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [30,90,180])
        #slightly modified
        # The numbers are directly copied and pasted from EasyGA results
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-360, -345.01740254474504, -326.96133522775204])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-326.96133522775204, -51.46618784411413, 0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [0, 72.1290447974928, 121.0773682633873])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0, 72.1290447974928, 121.0773682633873])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [121.0773682633873, 202.52891604785012, 360])
        
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1 ,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1]) 
        
                
        #Declare each fuzzy rule
        #rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        #rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))   
        rule6 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))    
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        #rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule11 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule14 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
     
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
     
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        
        #our code        
        ship_thrust = ctrl.Consequent(np.arange(-480.0, 480.0, 1.0), 'ship_thrust')
        distance_diff = ctrl.Antecedent(np.arange(-1000.0, 1000.0, 1), 'distance_diff')
        
        
        def generate_dist(low_limit, high_limit):
            NL_high = random.randint(low_limit, random.randint(-500, -100))
            NL_mid = random.randint(low_limit, NL_high)
            NS_mid = random.randint(NL_high, 0)
            PS_high = random.randint(0, random.randint(100, 500))
            PS_mid = random.randint(0, PS_high)
            PL_mid = random.randint(PS_high, high_limit)
            chromlist = [
                [low_limit, NL_mid, NL_high], 
                [NL_high, NS_mid, 0], 
                [0, PS_mid, PS_high],
                [PS_high, PL_mid, high_limit]
            ]
            return chromlist

    
        def generate_thrust(low_limit, high_limit):
            NL_high = random.randint(low_limit, random.randint(-7, -5))
            NL_mid = random.randint(low_limit, NL_high)
            NS_mid = random.randint(NL_high, 0)
            PS_high = random.randint(0, random.randint(4, 6))
            PS_mid = random.randint(0, PS_high)
            PL_mid = random.randint(PS_high, high_limit)
            chromlist = [
                [low_limit, NL_mid, NL_high], 
                [NL_high, NS_mid, 0], 
                [0, PS_mid, PS_high],
                [PS_high, PL_mid, high_limit]
            ]
            return chromlist
    
    
        def generate_chromosome():
            chromosome = {
                "distance_diff": generate_dist(-1000, 1000),
                "ship_thrust": generate_thrust(-150, 100)
            }
            return chromosome
    
        chromosome = generate_chromosome()
        # Membership functions for distance_diff
        distance_diff['NL'] = fuzz.trimf(distance_diff.universe, chromosome['distance_diff'][0])
        distance_diff['NS'] = fuzz.trimf(distance_diff.universe, chromosome['distance_diff'][1])
        #distance_diff['OK'] = fuzz.trimf(distance_diff.universe, [-100.0, 0.0, 100.0])
        distance_diff['PL'] = fuzz.trimf(distance_diff.universe, chromosome['distance_diff'][2])
        distance_diff['PS'] = fuzz.trimf(distance_diff.universe, chromosome['distance_diff'][3])

        ship_thrust['NL'] = fuzz.trimf(ship_thrust.universe, chromosome['ship_thrust'][0])
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, chromosome['ship_thrust'][1])
        #ship_thrust['NXS'] = fuzz.trimf(ship_thrust.universe, chromosome['ship_thrust'][2])
        ship_thrust['Zero'] = fuzz.trimf(ship_thrust.universe, [0 , 0, 0])
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, chromosome['ship_thrust'][2])
        ship_thrust['PL'] = fuzz.trimf(ship_thrust.universe, chromosome['ship_thrust'][3])

        #thrust_rule1 = ctrl.Rule(distance_diff['NL'], ship_thrust['NS'])
        thrust_rule1 = ctrl.Rule(distance_diff['NL'], ship_thrust['Zero'])
        thrust_rule2 = ctrl.Rule(distance_diff['NS'], ship_thrust['NL'])
        thrust_rule3 = ctrl.Rule(distance_diff['PS'], ship_thrust['PS'])
        thrust_rule4 = ctrl.Rule(distance_diff['PL'], ship_thrust['PL'])
        #thrust_rule5 = ctrl.Rule(distance_diff['OK'], ship_thrust['Zero'])
        
        # Create the fuzzy control system
        self.thrust_control_system = ctrl.ControlSystem([thrust_rule1, thrust_rule2, thrust_rule3, thrust_rule4])
        
        
        
        #Rules
        #Distance large or small 
        #velocity in 4 steps so consider 4 rules for the velocity steps
        
        

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.
        
        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    #closest_asteroid should dictate the thrust
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t
        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()
        #Our code
        closest_asteroid_radius = closest_asteroid["aster"]["radius"]
        closest_dist = math.sqrt(asteroid_ship_x**2 + asteroid_ship_y**2)
        #print(closest_asteroid["aster"]["velocity"])

        distance_diff_value = closest_dist - (bullet_t * math.sqrt((closest_asteroid["aster"]["velocity"][0]**2) + (closest_asteroid["aster"]["velocity"][1]**2)))
        distance_diff_value = closest_dist - 10*closest_asteroid_radius
        # print(distance_diff_value)
        
        # Create the simulation object
        thrust_simulation = ctrl.ControlSystemSimulation(self.thrust_control_system)
        
        thrust_simulation.input['distance_diff'] = distance_diff_value
        
        # Computing the thrust
        thrust_simulation.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        # Get the defuzzified output
        thrust = thrust_simulation.output['ship_thrust']
        
        self.eval_frames +=1

        if ship_state["is_respawning"] == True:
            fire = False
        
        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        drop_mine = False
        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "My Controller"
    


# def fitness(chromosome):
#     for i in chromosome:
#         chromosomes = i.value
#     my_test_scenario = Scenario(name='Test Scenario',
#     num_asteroids=5,
#     ship_states=[
#     {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1},
#     {'position': (600, 400), 'angle': 90, 'lives': 3, 'team': 2},
#     ],

#     map_size=(1000, 800),
#     time_limit=60,
#     ammo_limit_multiplier=0,
#     stop_if_no_ammo=False)
#     game_settings = {'perf_tracker': True,
#     'graphics_type': GraphicsType.Tkinter,
#     'realtime_multiplier': 1,
#     'graphics_obj': None}
#     my_controller = 0
#     test_controller = 0
#     for i in range(2):
#         game = TrainerEnvironment(settings=game_settings) # Use this for max-speed, no-graphics simulation
#         score, perf_data = game.run(scenario=my_test_scenario, controllers = [MyController(chromosomes), ScottDickController()])
#         if ([controller.eval_frames for controller in score.final_controllers][0] > 300):
#             my_controller += [team.asteroids_hit for team in score.teams][0]
#             test_controller += 1
#     print(my_controller / test_controller)
#     return (my_controller / test_controller)

# def GAtrain_fitness():
#     ga = EasyGA.GA()
#     ga.gene_impl = lambda: generate_chromosome()
#     ga.chromosome_length = 1
#     ga.population_size = 20
#     ga.target_fitness_type = 'max'
#     ga.generation_goal = 1
#     # ga.crossover_probability=0.8,
#     # ga.mutation_probability=0.2,
#     ga.fitness_function_impl = fitness
#     ga.evolve()
#     ga.print_best_chromosome()
    
# GAtrain_fitness()

