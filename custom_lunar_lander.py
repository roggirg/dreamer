import random
import sys, math
import time

import numpy as np

import Box2D
import pygame
from pygame.locals import VIDEORESIZE
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
from gym.utils.play import display_arr

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder
VIEWPORT_W = 600
VIEWPORT_H = 400

ORIG_CONFIG = {
    0: {
        'LANDER_POLY': [(-25, +30), (-30, 0), (-30, -15), (+30, -15), (+30, 0), (+25, +30)],
        'LEG_AWAY': 28,
        'LEG_DOWN': 26,
        'LEG_W': 5,
        'LEG_H': 18,
        'LEG_SPRING_TORQUE': 120,
        'MAIN_ENGINE_POWER': 13.0,
        'SIDE_ENGINE_POWER': 0.6,
        'SIDE_ENGINE_HEIGHT': 14.0,
        'SIDE_ENGINE_AWAY': 12.0,
        'LANDER_COLOR_1': (0.5, 0.4, 0.9),
        'LANDER_COLOR_2': (0.3, 0.3, 0.5),
        'SKY_COLOR': (0.0, 0.0, 0.0),
        'BALL_COLOR': (0.5, 0.4, 0.5),
        'MAX_M_POW': 5.0,
        'MAX_S_POW': 10.0
    }
}

CONFIGS = {
    0: {
        'LANDER_POLY': [(-25, +30), (-30, 0), (-30, -15), (+30, -15), (+30, 0), (+25, +30)],
        'LEG_AWAY': 28,
        'LEG_DOWN': 26,
        'LEG_W': 5,
        'LEG_H': 18,
        'LEG_SPRING_TORQUE': 120,
        'MAIN_ENGINE_POWER': 65.0,
        'SIDE_ENGINE_POWER': 6.0,
        'SIDE_ENGINE_HEIGHT': 25.0,
        'SIDE_ENGINE_AWAY': 20.0,
        'LANDER_COLOR_1': (0.5, 0.4, 0.9),
        'LANDER_COLOR_2': (0.3, 0.3, 0.5),
        'SKY_COLOR': (0.0, 0.0, 0.0),
        'BALL_COLOR': (0.5, 0.4, 0.5),
        'MAX_M_POW': 1.0,
        'MAX_S_POW': 1.0
    },
    1: {
        'LANDER_POLY': [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)],
        'LEG_AWAY': 20,
        'LEG_DOWN': 18,
        'LEG_W': 2,
        'LEG_H': 8,
        'LEG_SPRING_TORQUE': 40,
        'MAIN_ENGINE_POWER': 13,
        'SIDE_ENGINE_POWER': 0.6,
        'SIDE_ENGINE_HEIGHT': 14.0,
        'SIDE_ENGINE_AWAY': 12.0,
        'LANDER_COLOR_1': (0.1, 0.7, 0.3),
        'LANDER_COLOR_2': (0.5, 0.2, 0.9),
        'SKY_COLOR': (0.9, 0.1, 0.2),
        'BALL_COLOR': (0.2, 0.7, 0.9),
        'MAX_M_POW': 1.0,
        'MAX_S_POW': 1.0
    },
    2: {
        'LANDER_POLY': [(-20, +25), (-25, 0), (-25, -16), (+25, -16), (+25, 0), (+20, +25)],
        'LEG_AWAY': 24,
        'LEG_DOWN': 24,
        'LEG_W': 3.5,
        'LEG_H': 14,
        'LEG_SPRING_TORQUE': 80,
        'MAIN_ENGINE_POWER': 35.0,
        'SIDE_ENGINE_POWER': 3.5,
        'SIDE_ENGINE_HEIGHT': 20.0,
        'SIDE_ENGINE_AWAY': 18.0,
        'LANDER_COLOR_1': (0.9, 0.7, 0.4),
        'LANDER_COLOR_2': (0.5, 0.8, 0.3),
        'SKY_COLOR': (0.3, 0.2, 0.8),
        'BALL_COLOR': (0.6, 0.7, 0.1),
        'MAX_M_POW': 1.0,
        'MAX_S_POW': 1.0
    },
    3: {
        'LANDER_POLY': [(-17, +21), (-21, 0), (-21, -7), (+21, -7), (+21, 0), (+17, +21)],
        'LEG_AWAY': 22,
        'LEG_DOWN': 20,
        'LEG_W': 3,
        'LEG_H': 12,
        'LEG_SPRING_TORQUE': 60,
        'MAIN_ENGINE_POWER': 25.0,
        'SIDE_ENGINE_POWER': 2.5,
        'SIDE_ENGINE_HEIGHT': 17.0,
        'SIDE_ENGINE_AWAY': 16.0,
        'LANDER_COLOR_1': (0.8, 0.2, 0.1),
        'LANDER_COLOR_2': (0.7, 0.25, 0.3),
        'SKY_COLOR': (0.1, 0.8, 0.7),
        'BALL_COLOR': (0.4, 0.2, 0.4),
        'MAX_M_POW': 1.0,
        'MAX_S_POW': 1.0
    },
    4: {
        'LANDER_POLY': [(-29, +35), (-35, 0), (-35, -18), (+35, -18), (+35, 0), (+29, +35)],
        'LEG_AWAY': 32,
        'LEG_DOWN': 29,
        'LEG_W': 7,
        'LEG_H': 20,
        'LEG_SPRING_TORQUE': 180,
        'MAIN_ENGINE_POWER': 80.0,
        'SIDE_ENGINE_POWER': 8.0,
        'SIDE_ENGINE_HEIGHT': 29.0,
        'SIDE_ENGINE_AWAY': 26.0,
        'LANDER_COLOR_1': (0.75, 0.1, 0.95),
        'LANDER_COLOR_2': (0.5, 0.2, 0.8),
        'SKY_COLOR': (0.2, 0.6, 0.2),
        'BALL_COLOR': (0.8, 0.3, 0.9),
        'MAX_M_POW': 1.0,
        'MAX_S_POW': 1.0
    }
}


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander in [contact.fixtureA.body, contact.fixtureB.body] and \
                self.env.moon in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = False
    put_balls = False
    orig_config = True
    fixed_bg = False
    use_alt_reward = False
    custom_state = True
    inv_data = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.balls = []

        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        # Uncomment the following to fix background
        if self.fixed_bg:
            H = VIEWPORT_H / SCALE
            self.height = np.array([0.1, 0.6, 0.2, 0.8, 0.1, 0.7, 0.6, 0.7, 0.6, 0.9, 0.3, 0.6]) * (H / 2)

        self.max_episode_length = 1000

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self._clean_balls(True)

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        if self.orig_config:
            self.config = ORIG_CONFIG[0]
        else:
            self.config = CONFIGS[random.randint(0, 4)]

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # terrain
        CHUNKS = 11
        if not self.fixed_bg:
            self.height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        self.height[CHUNKS // 2 - 2] = self.helipad_y
        self.height[CHUNKS // 2 - 1] = self.helipad_y
        self.height[CHUNKS // 2 + 0] = self.helipad_y
        self.height[CHUNKS // 2 + 1] = self.helipad_y
        self.height[CHUNKS // 2 + 2] = self.helipad_y
        self.smooth_y = [0.33 * (self.height[i - 1] + self.height[i + 0] + self.height[i + 1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], self.smooth_y[i])
            p2 = (chunk_x[i + 1], self.smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        if not self.inv_data:
            initial_x = VIEWPORT_W / SCALE / 2
            initial_y = VIEWPORT_H / SCALE
        else:
            initial_x = self.np_random.uniform(0.1, 0.9) * VIEWPORT_W / SCALE
            initial_y = self.np_random.uniform(0.40, 1) * VIEWPORT_H / SCALE

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in self.config['LANDER_POLY']]),
                density=5.0,
                friction=0.1,
                groupIndex=1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = self.config['LANDER_COLOR_1']
        self.lander.color2 = self.config['LANDER_COLOR_2']
        if not self.inv_data:
            self.lander.ApplyForceToCenter((
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * self.config['LEG_AWAY'] / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(self.config['LEG_W'] / SCALE, self.config['LEG_H'] / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    groupIndex=1,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = self.config['LANDER_COLOR_1']
            leg.color2 = self.config['LANDER_COLOR_2']
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * self.config['LEG_AWAY'] / SCALE, self.config['LEG_DOWN'] / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.config['LEG_SPRING_TORQUE'],
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        if self.put_balls:
            num_balls = 3
            for i in range(num_balls):
                radius = random.randint(10, 30)
                mass = random.uniform(2.5, 10.5)
                y_pos = random.uniform(0.35, 0.9) * VIEWPORT_H / SCALE
                x_pos = random.uniform(0.9, 1.0) * VIEWPORT_W / SCALE
                impulse_strength = (random.randint(-200, -50), random.randint(0.0, 100.0))
                p = self._create_ball(mass, x_pos, y_pos, ttl=1.0, radius=radius)
                impulse_pos = (p.position[0], p.position[1])
                p.ApplyLinearImpulse(impulse_strength, impulse_pos, True)
                p.color1 = self.config['BALL_COLOR']
                p.color2 = self.config['BALL_COLOR']

        self.t = 0

        self.drawlist = [self.lander] + self.legs + self.balls
        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _create_ball(self, mass, x, y, ttl, radius):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                groupIndex=1,
                categoryBits=0x0300,
                maskBits=0x001,  # collide only with ground
                restitution=0.5)
        )
        p.ttl = ttl
        self.balls.append(p)
        self._clean_balls(False)
        return p

    def _clean_balls(self, all):
        while self.balls and (all or self.balls[0].ttl < 0):
            self.world.DestroyBody(self.balls.pop(0))

    def _clean_balls_if_static(self):
        for (i, ball) in enumerate(self.balls):
            if ball.position[1] == 0 and ball.linearVelocity.y == 0:
                self.world.DestroyBody(self.balls.pop(i))

    def step(self, action):
        self.t += 1
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            max_m_power = self.config['MAX_M_POW']
            if self.continuous:
                m_power = (max_m_power*np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= max_m_power
            else:
                m_power = 1.0 * max_m_power
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[
                1]  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            # p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1],
            #                           m_power)  # particles are just a decoration, 3.5 is here to make particle speed adequate
            # p.ApplyLinearImpulse((ox * self.config['MAIN_ENGINE_POWER'] * m_power, oy * self.config['MAIN_ENGINE_POWER']
            #                       * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * self.config['MAIN_ENGINE_POWER'] * m_power,
                                            -oy * self.config['MAIN_ENGINE_POWER'] * m_power), impulse_pos, True)
            m_power /= max_m_power

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            max_s_power = self.config['MAX_S_POW']
            if self.continuous:
                direction = np.sign(action[1])
                s_power = max_s_power*np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= max_s_power
            else:
                direction = action - 2
                s_power = 1.0 * max_s_power
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * self.config['SIDE_ENGINE_AWAY'] / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * self.config['SIDE_ENGINE_AWAY'] / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * self.config['SIDE_ENGINE_HEIGHT'] / SCALE)
            # p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            # p.ApplyLinearImpulse((ox * self.config['SIDE_ENGINE_HEIGHT'] * s_power,
            #                       oy * self.config['SIDE_ENGINE_HEIGHT'] * s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * self.config['SIDE_ENGINE_POWER'] * s_power,
                                            -oy * self.config['SIDE_ENGINE_POWER'] * s_power), impulse_pos, True)
            s_power /= max_s_power

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + self.config['LEG_DOWN'] / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]
        assert len(state) == 8

        if self.custom_state:
            leg_heights = self.get_leg_heights()
            state = [
                (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
                (pos.y - (self.helipad_y + self.config['LEG_DOWN'] / SCALE)) / (VIEWPORT_H / SCALE / 2),
                vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
                self.lander.angle,
                20.0 * self.lander.angularVelocity / FPS,
                leg_heights[0],
                leg_heights[1]
            ]

        if self.t % 150 == 0:
            # Remove balls after 200 timesteps.
            if self.put_balls:
                self._clean_balls(True)
                num_balls = 3
                for i in range(num_balls):
                    radius = random.randint(10, 30)
                    mass = random.uniform(2.5, 10.5)
                    y_pos = random.uniform(0.35, 0.9) * VIEWPORT_H / SCALE
                    x_pos = random.uniform(0.9, 1.0) * VIEWPORT_W / SCALE
                    impulse_strength = (random.randint(-200, -50), random.randint(0.0, 100.0))
                    p = self._create_ball(mass, x_pos, y_pos, ttl=1.0, radius=radius)
                    impulse_pos = (p.position[0], p.position[1])
                    p.ApplyLinearImpulse(impulse_strength, impulse_pos, True)
                    p.color1 = (random.uniform(0.0, 0.45), random.uniform(0.45, 1.0), random.uniform(0.0, 0.85))
                    p.color2 = (random.uniform(0.35, 1.0), random.uniform(0.35, 1.0), random.uniform(0.55, 1.0))

        if self.use_alt_reward:
            reward, done = self.alt_reward_f()
            if reward == 0:
                # If we didn't terminate yet, less fuel spent is better, about -30 for heurisic landing
                reward -= m_power * 0.30
                reward -= s_power * 0.03
        else:  # Original reward function
            reward = 0
            shaping = \
                - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
                - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
                - 100 * abs(state[4]) + \
                10 * (1.0 if self.legs[0].ground_contact else 0.0) + \
                10 * (1.0 if self.legs[1].ground_contact else 0.0)

            # And ten points for legs contact, the idea is if you lose contact again after landing,
            # you get negative reward.
            if self.prev_shaping is not None:
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping

            reward -= m_power * 0.30
            reward -= s_power * 0.03

            done = False
            if self.game_over or abs(state[0]) >= 1.0:
                done = True
                reward = -100
            if not self.lander.awake:
                done = True
                reward = +100

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=self.config['SKY_COLOR'])

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    # self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def alt_reward_f(self):
        reward = 0
        done = False
        if self.t == self.max_episode_length:
            done = True
            if self.legs[0].ground_contact or self.legs[1].ground_contact:  # on the ground.
                if self.legs[0].position.x <= self.helipad_x2 and self.legs[1].position.x >= self.helipad_x1:
                    # landed in the landing zone.
                    reward = 100
                else:  # Not in the landing zone.
                    reward = 50
            else:  # no-land scenario
                reward = -100
        elif self.game_over or abs((self.lander.position.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)) >= 1.0:
            reward = -100
            done = True
        elif not self.lander.awake:  # landed
            done = True
            if self.legs[0].ground_contact or self.legs[1].ground_contact:  # on the ground.
                if self.legs[0].position.x <= self.helipad_x2 and \
                        self.legs[1].position.x >= self.helipad_x1:
                    # landed in the landing zone.
                    reward = 100
                else:  # Not in the landing zone.
                    reward = 50

        return reward, done

    def get_leg_heights(self):
        # Configured for CHUNKS=11
        leg_heights = []
        for i in range(len(self.legs)):
            x_leg_pos = int(self.legs[i].position.x // 2)
            x_leg_prop = self.legs[i].position.x / 2
            if x_leg_pos < 10 and x_leg_pos >= 0:
                leg_heights.append(self.legs[i].position.y - (
                        (x_leg_prop - x_leg_pos) * (self.smooth_y[x_leg_pos + 1] - self.smooth_y[x_leg_pos]) +
                        self.smooth_y[x_leg_pos]
                ))
            elif x_leg_pos == 10:
                leg_heights.append(self.legs[i].position.y - self.smooth_y[x_leg_pos])
            else:
                leg_heights.append(self.legs[i].position.y - self.smooth_y[0])
            leg_heights[i] /= (VIEWPORT_H / SCALE)

        return leg_heights


class LunarLanderContinuous(LunarLander):
    continuous = True


class LunarLanderContinuousFixedBackground(LunarLander):
    continuous = True
    fixed_bg = True


class LunarLanderContinuousAltReward(LunarLander):
    continuous = True
    use_alt_reward = True


class LunarLanderContinuousFixedBackgroundAltReward(LunarLander):
    continuous = True
    fixed_bg = True
    use_alt_reward = True


class LunarLanderContinuousBalls(LunarLander):
    continuous = True
    put_balls = True


class LunarLanderContinuousNewConfigs(LunarLander):
    continuous = True
    orig_config = False


class LunarLanderContinuousBallsNewConfigs(LunarLander):
    continuous = True
    orig_config = False
    put_balls = True


class LunarLanderInverseModel(LunarLander):
    inv_data = True


class LunarLanderContinuousInverseModel(LunarLander):
    inv_data = True
    continuous = True


def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])  # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
    # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    # if s[6] or s[7]:  # legs have contact
    #     angle_todo = 0
    #     hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        # a = np.array([hover_todo * 100 - 1, -angle_todo * 200])
        # a = np.clip(a, -10, +10)
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False, action_repeat=1):
    env.seed(seed)
    rewards = []
    for i in range(2):
        total_reward = 0
        steps = 0
        s = env.reset()
        # x_rand_pos = random.uniform(-0.5, 0.5)
        # y_rand_pos = random.uniform(-0.5, 0.5)
        while True:
            # a = [0, 0]
            a = heuristic(env, s)  # + [0, 0.8*random.uniform(-1, 1)]  # 2nd term adds noise to avoid successful done.
            for _ in range(action_repeat):
                s, r, done, info = env.step(a)
            s[0] += 0.
            s[1] -= 0.0
            total_reward += r

            if render:
                still_open = env.render()
                if still_open == False:
                    break

            steps += 1
            time.sleep(0.01)
            if steps == 600:
                done = True
            if done:
                done_type = -1
                if steps == 600:
                    if env.legs[0].ground_contact or env.legs[1].ground_contact:  # on the ground.
                        if env.legs[0].position.x <= env.helipad_x2 and env.legs[1].position.x >= env.helipad_x1:
                            # landed in the landing zone.
                            done_type = 0
                        else:  # Not in the landing zone.
                            done_type = 1
                    else:  # no-land scenario
                        done_type = 3
                elif env.game_over:
                    done_type = 2
                if not env.lander.awake:
                    if env.legs[0].ground_contact or env.legs[1].ground_contact:  # on the ground.
                        if env.legs[0].position.x <= env.helipad_x2 and \
                                env.legs[1].position.x >= env.helipad_x1:
                            # landed in the landing zone.
                            done_type = 0
                        else:  # Not in the landing zone.
                            done_type = 1
                print(done_type)
                break
        rewards.append(total_reward)
    print(np.mean(rewards))


def play(env, transpose=True, fps=10, zoom=None, callback=None, keys_to_action=None, action_repeat=1):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    env.reset()
    rendered = env.render(mode='rgb_array')

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True
    action = None

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            print("Env Time steps", env.t)
            print("Velocities:", obs[2], obs[3])
            prev_action = action
            action = keys_to_action.get(tuple(sorted(pressed_keys)), None)
            if action == prev_action:
                action = None
                prev_action = action
            if action is not None:
                for i in range(action_repeat):
                    prev_obs = obs
                    obs, rew, env_done, info = env.step(action)
                    if callback is not None:
                        callback(prev_obs, obs, action, rew, env_done, info)
            print("Velocities:", obs[2], obs[3])
            obs[2], obs[3] = 0, 0
        if obs is not None:
            rendered = env.render(mode='rgb_array')
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


def manual_control(env, action_repeat=1):
    # from gym.utils.play import play
    keys_to_action = {
        (ord(' '),): 0,
        (ord('w'),): 2,
        (ord('a'),): 1,
        (ord('d'),): 3,
    }
    play(env=env, keys_to_action=keys_to_action, action_repeat=action_repeat)


if __name__ == '__main__':
    demo_heuristic_lander(LunarLander(), render=True, action_repeat=2)
    # manual_control(LunarLander(), action_repeat=2)


