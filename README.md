# actor_critic

actor_critic.py contains a proper actor critic with the critic feeding a calculation of the td-error to the actor.

actor_critic_sorta.py contains an algorithm that looks like an actor critic, it has the same architecture, but the critic does not feed anything to the actor. Instead, the actor uses the policy gradient theorm with a value function baseline (i.e. the advantage) for its loss.
