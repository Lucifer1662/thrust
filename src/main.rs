extern crate fps_counter;
extern crate multimap;
extern crate nalgebra;
extern crate piston_window;
use rand::Rng;

use std::ops::Add;
use std::option::Option;

use std::{sync::atomic::AtomicI32, sync::atomic::Ordering};

use nalgebra::*;
use piston_window::*;

pub type Vec2 = Vector2<f64>;

type ReactFunc = Option<fn(atom: &Atom) -> std::vec::Vec<Atom>>;

#[derive(Copy, Clone)]
struct Atom {
    pos: Vec2,
    velocity: Vec2,
    impulse_velocity: Vec2,
    acceleration: Vec2,
    radius: f64,
    id: i32,
    rotation: f64,
    canExplode: bool,
    mass: f64,
    react: ReactFunc,
}

static NEXT_ID: AtomicI32 = AtomicI32::new(0);

struct Collision {
    dir: Vec2,
    depth: f64,
    relative_velocity: Vec2,
    mass_coefficient: f64,
}

// fn reflect(i: Vec2, n: Vec2) -> Vec2 {
//     i - 2.0 * (i.dot(&n)) * n
// }

fn react(a: &Atom) -> std::vec::Vec<Atom> {
    let mut rng = rand::thread_rng();

    let mut top = a.pos;
    top.y += a.radius / 2.0;
    let mut bottom = a.pos;
    bottom.y -= a.radius / 2.0;
    let mut atoms = vec![
        Atom::new(top, a.mass / 2.0, a.radius / 2.0, None),
        Atom::new(bottom, a.mass / 2.0, a.radius / 2.0, None),
    ];
    atoms[0].velocity = Vec2::new(rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0));
    atoms[1].velocity = -atoms[0].velocity;
    atoms
}

impl Atom {
    fn new(pos: Vec2, mass: f64, radius: f64, react: ReactFunc) -> Atom {
        Atom {
            acceleration: Vec2::new(0.0, 0.0),
            velocity: Vec2::new(0.0, 0.0),
            impulse_velocity: Vec2::zeros(),
            pos: pos,
            radius: radius,
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            rotation: 0.0,
            canExplode: true,
            mass: mass,
            react: react,
        }
    }

    fn update(&mut self, dt: f64) {
        self.velocity += self.acceleration * dt;
        self.pos += self.velocity * dt + self.impulse_velocity;
        self.impulse_velocity = Vec2::zeros();
    }

    fn resolve(&mut self, collisions: &[Collision]) {
        collisions.iter().for_each(|collision| {
            let n = collision.dir;

            let d = collision.depth;
            self.impulse_velocity = n * d;

            // Calculate relative velocity

            // Calculate relative velocity in terms of the normal direction
            let vel_along_normal = collision.relative_velocity.dot(&n);

            // if self.velocity.dot(&n) < 0.0 {
            //     self.velocity = reflect(self.velocity, n) * 0.8;
            // }

            // Do not resolve if velocities are separating
            if vel_along_normal <= 0.0 {
                // Calculate restitution
                let e = 0.8;

                // Calculate impulse scalar
                let j = (-(1.0 + e) * vel_along_normal) / collision.mass_coefficient;

                // Apply impulse
                let impulse = j * n;
                self.velocity += 1.0 / self.mass * impulse;
            }
        })
    }

    fn will_explode(&self) -> bool {
        let mut rng = rand::thread_rng();
        rng.gen_bool(0.001)
    }

    fn chemical_update(&self) -> std::vec::Vec<Atom> {
        if self.react.is_some() {
            self.react.unwrap()(self)
        } else {
            vec![]
        }
    }
}

fn collide(a1: &Atom, a2: &Atom) -> Option<Collision> {
    let dif = a1.pos - a2.pos;
    let len = dif.magnitude();
    if len < a1.radius + a2.radius {
        let s = Some(Collision {
            dir: dif.normalize(),
            depth: ((a1.radius + a2.radius) - len).max(0.00001),
            relative_velocity: a1.velocity - a2.velocity,
            mass_coefficient: (1.0 / a1.mass) + (1.0 / a2.mass),
        });
        return s;
    } else {
        None
    }
}

struct LineSegment {
    start: Vec2,
    end: Vec2,
}

fn collide_line(a1: &Atom, l: &LineSegment) -> Option<Collision> {
    let p = a1.pos;
    let m = l.end - l.start;
    let pa = p - l.start;
    let t = (pa.dot(&m) / m.dot(&m)).min(1.0).max(0.0);

    let closest_line = l.start + t * m;
    let dif = closest_line - p;
    let dist = dif.magnitude();
    let travel_dist = dif.normalize().dot(&(a1.velocity * DT)).max(0.0);

    if dist < travel_dist + a1.radius {
        Some(Collision {
            dir: -dif.normalize(),
            depth: a1.radius - dist,
            relative_velocity: a1.velocity,
            mass_coefficient: (1.0 / a1.mass),
        })
    } else {
        None
    }
}

static DT: f64 = 0.1;

fn to_space_key(pos: &Vec2) -> (i64, i64) {
    let s = ((pos.x as i64) / 6, (pos.y as i64) / 6);
    s
}

fn add_tup<T1, T2>(x: (T1, T2), y: (T1, T2)) -> (T1, T2)
where
    T1: Add<T1, Output = T1>,
    T2: Add<T2, Output = T2>,
{
    (x.0 + y.0, x.1 + y.1)
}

fn collision_space(
    atom: &Atom,
    atoms: &[Atom],
    space: &multimap::MultiMap<(i64, i64), usize>,
    lines: &[LineSegment],
) -> std::vec::Vec<Collision> {
    let neighbors: std::vec::Vec<(i64, i64)> = vec![
        (0, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (1, 0),
        (-1, 0),
        (0, -1),
    ];

    let mut s: std::vec::Vec<Collision> = neighbors
        .iter()
        .flat_map(|p| {
            let s: std::vec::Vec<Collision> = space
                .get_vec(&add_tup(to_space_key(&atom.pos), *p))
                .unwrap_or(&std::vec::Vec::new())
                .iter()
                .map(|i| atoms.get(*i).unwrap())
                .filter(|atom2| atom2.id != atom.id)
                .filter_map(|atom2| collide(atom, atom2))
                .map(|f| f)
                .collect();
            s
        })
        .collect();

    s.extend(lines.iter().filter_map(|l| collide_line(atom, l)));
    s
}

fn main() {
    let mut atoms = vec![
        // Atom::new(Vec2::new(50.0, 100.0), 2.0),
        // Atom::new(Vec2::new(50.0, 50.0), 1.0, 2.5, None),
        // Atom::new(Vec2::new(50.0, 50.0), 1.0, 5.0,  Some(react)),
    ];

    for x in 0..30 {
        for y in 0..30 {
            atoms.push(Atom::new(
                Vec2::new(200.0 + (x as f64) * 5.0f64, 300.0 + (y as f64) * 5.0f64),
                1.0,
                5.0,
                Some(react),
            ));
        }
    }

    let lines = vec![
        LineSegment {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(1000.0, 0.0),
        },
        LineSegment {
            start: Vec2::new(0.0, 400.0),
            end: Vec2::new(1000.0, 400.0),
        },
        LineSegment {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(0.0, 400.0),
        },
        LineSegment {
            start: Vec2::new(600.0, 0.0),
            end: Vec2::new(600.0, 400.0),
        },
        LineSegment {
            start: Vec2::new(100.0, 300.0),
            end: Vec2::new(600.0, 0.0),
        },
    ];

    let mut window: PistonWindow = WindowSettings::new("Hello Piston!", [640, 480])
        .exit_on_esc(true)
        .build()
        .unwrap();

    let mut frames = 0;
    let mut passed = 0.0;

    while let Some(event) = window.next() {
        if let Some(_) = event.render_args() {
            window.draw_2d(&event, |context, graphics, _device| {
                let transform = context.transform.flip_v().trans(0.0, -400.0);

                clear([1.0; 4], graphics);

                lines.iter().for_each(|l| {
                    line(
                        color::BLUE,
                        2.0,
                        [l.start.x, l.start.y, l.end.x, l.end.y],
                        transform,
                        graphics,
                    );
                });

                for i in 0..atoms.len() {
                    let a = atoms.get(i).unwrap();
                    rectangle(
                        color::NAVY,
                        rectangle::centered_square(0.0, 0.0, a.radius),
                        transform.trans(a.pos.x, a.pos.y),
                        // .rot_rad(a.rotation),
                        graphics,
                    );
                }
            });
        }

        if let Some(u) = event.update_args() {
            frames += 1;

            let mut space: multimap::MultiMap<(i64, i64), usize> = multimap::MultiMap::new();

            for i in 0..atoms.len() {
                let a = atoms.get(i).unwrap();
                space.insert(to_space_key(&a.pos), i)
            }

            for i in 0..atoms.len() {
                let a = atoms.get(i).unwrap();

                let cols = collision_space(a, &atoms, &space, &lines);

                let a = atoms.get_mut(i).unwrap();
                a.resolve(&cols);
            }

            atoms.iter_mut().for_each(|atom| atom.update(DT));

            for i in 0..atoms.len() {
                let a = atoms.get_mut(i).unwrap();
                if a.will_explode() {
                    let new_atoms = a.chemical_update();
                    if new_atoms.len() > 0 {
                        let (first, rest) = new_atoms.split_first().unwrap();
                        *a = *first;
                        atoms.extend(rest);
                    }
                }
            }

            passed += u.dt;

            if passed > 1.0 {
                let fps = (frames as f64) / passed;

                println!("FPS: {}", fps);

                frames = 0;
                passed = 0.0;
            }
        }
    }
}
