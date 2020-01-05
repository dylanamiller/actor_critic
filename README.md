# actor_critic

actor_critic.py contains a proper actor critic with the critic feeding a calculation of the td-error to the actor.

actor_critic_sorta.py contains an algorithm that looks like an actor critic; it has the same architecture, but the critic does not feed anything to the actor. Instead, the actor uses the policy gradient theorm with a value function baseline (i.e. the advantage) for its loss. Some people may still call this an actor critic, and they wouldn't be wrong, but given the relationship between the actor and the critic, it may be closer to REINFORCE with a baseline with the additional calculation of a value function. It does not really act like a critic (i.e. it does not used for bootstrapping in the algorihm). See (page 273, the first paragraph of section 13.5)[http://www.incompleteideas.net/book/bookdraft2017nov5.pdf].
