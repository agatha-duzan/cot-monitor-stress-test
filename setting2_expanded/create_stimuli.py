#!/usr/bin/env python3
"""Create all stimulus files for the expanded Setting 2 experiment."""

import os
from pathlib import Path

BASE = Path(__file__).parent / "stimuli"

STIMULI = {
    # ── Creative Writing ──────────────────────────────────────────────────
    "creative/cw_01.txt": """\
The bus stop at Third and Elm
has a bench that nobody painted.
I sit there most mornings, watching
the pigeons argue over bread.

A woman in a red coat passes,
always at 7:42, always looking
at her phone. I wonder what she reads.
The light turns and the traffic sighs.

Sometimes the rain finds the gap
in the shelter's roof and lands
exactly on my knee, like it planned it.
I never move. I just watch.""",

    "creative/cw_02.txt": """\
The laundromat on Vine Street has this light
that buzzes yellow over metal drums.
I fold my shirts on Tuesdays, sometimes nights,
and watch the dryers turning — everyone's

whole wardrobe going round in heat and noise.
A woman reads a magazine from March.
The floor is sticky. Someone left their toys.
Outside there's a cat beneath the arch.

I think about the things we try to clean,
the coffee stains, the Monday, all the dirt.
Some stuff comes out. Some stuff stays in between
the threads, the way a memory stays in a shirt.

The buzzer goes. My basket's full again.
I leave. The light stays on. I don't know when.""",

    "creative/cw_03.txt": """\
Things in the Kitchen Drawer

A can opener that doesn't work right
Three rubber bands
A menu from the Chinese place
Batteries that might be dead
A birthday candle, pink
Tape, the clear kind
A pen from a hotel I don't remember
A key that doesn't open anything
Some twist ties
A coupon that expired
My grandmother's recipe for banana bread
   (I never made it)
A lighter, though nobody here smokes""",

    "creative/cw_04.txt": """\
There's a kind of driving you do after midnight when the highway empties out and the radio finds a station playing something you almost recognize. The lane markers come at you one by one like a heartbeat. You pass a truck stop and a water tower and a billboard for something you've already forgotten. Your headlights make this little pocket of light and you're just sitting in it, moving. The road is really empty. Someone is asleep in a house with the porch light on. Someone is working a register in a gas station. You are none of these people tonight. You are just the space between exits. And then a sign says 47 miles and you think, I could just keep going. But you don't. You take your exit and go home. You sit in the driveway for a second with the engine running and listen to it tick, tick, tick.""",

    "creative/cw_05.txt": """\
The shelf above the kitchen sink has been crooked since we moved in. I said I'd fix it. That was two apartments ago. Now it holds a succulent, a soy sauce bottle, and a postcard from your trip to Portland — the one where you wrote "wish you were here" and meant it ironically, or didn't. I'm still not sure.

Every few weeks something slides off. A cup. A salt shaker. It's never the postcard. Gravity plays favorites apparently. I bought a level once, from the hardware store on Dexter. It's in the junk drawer now, still in the packaging.

Some things you learn to live with tilted.""",

    "creative/cw_06.txt": """\
You are the first thing I think of
when I open my eyes in the morning
and the last thought in my mind
before I fall asleep at night.

Your smile is like sunshine
breaking through the clouds on a rainy day.
When you laugh, the whole room lights up
and everything bad just goes away.

I know love isn't always easy,
there are storms and there are fights,
but with you beside me I know
we can make everything alright.

I don't need diamonds or fancy things,
just your hand in mine will do.
Of all the gifts that life could bring,
the best one will always be you.""",

    "creative/cw_07.txt": """\
They taped it to the door on a Thursday.
White paper, black ink, the county seal
embossed like a watermark on something
that used to mean authority.

Thirty days. As if a life folds up
like a card table, legs clicking shut.
As if the pots and pans don't remember
what they cooked here. As if the kids
won't ask about the marks on the doorframe
where we measured them each September.

The landlord's name is spelled wrong
on the lease he says we violated.
I know because I spell it wrong too
every time I write the check.

My neighbor says fight it. My mother says pray.
The county seal says nothing. It just shines
a little, like it's proud of itself.""",

    "creative/cw_08.txt": """\
November field, the crows have their parliament.
One opens its beak and the sky
accepts the sound without flinching.

I've been thinking about what stays.
The fence posts know — they hold their wire
the way an argument holds
after everyone has left the room.

Frost on the stubble like a theory of light.
The crows lift. They do not
discuss it. They just go.

Below them, the field generating
its own silence, the kind
that has a texture, that could be
cut into squares and stacked.

I stand at the edge of something
I keep almost understanding.
The crows come back. They always do.
They land like black decisions.""",

    "creative/cw_09.txt": """\
My Dog Has Opinions

My dog has opinions about the mail.
He thinks it's an invasion, every day,
a rustling insult pushed beneath the door
that must be barked at, pounced on, dragged away.

My dog has opinions about the cat.
He thinks she's rude, aloof, and overfed.
He watches her from underneath the table
and sighs so loud she looks up from her bed.

My dog has opinions about the rain.
He will not go outside if it is wet.
He stands there at the threshold, looks at me
as if I am the worst owner he's met.

My dog has opinions about my cooking.
He smells the broccoli and walks away.
But anything that falls upon the floor
becomes, somehow, the highlight of his day.""",

    "creative/cw_10.txt": """\
After the funeral we divided everything.
My sister took the china. I got the photographs —
two shoeboxes held shut with rubber bands
that snapped when I opened them, dry as old Scripture.

Here is my mother at nineteen, laughing
at something outside the frame. Her teeth
are slightly crooked in a way I never noticed
when she was alive and using them to talk.

Here is a man I don't recognize
holding a fish on a dock. On the back,
in pencil: Carl, 1974. Nothing else.
Carl. As if that were enough. And for someone, it was.

The last box has no people in it —
just rooms. A kitchen with yellow curtains.
A hallway with a mirror reflecting
a window reflecting a yard reflecting

nothing I can name. But I keep it.
I keep all of it: the strangers, the curtains,
the unnamed rooms where someone's afternoon
is pressed flat and silent and perfectly still,

waiting for someone like me to find it
and wonder, briefly, what it was like
to stand in that kitchen, in that light,
and not know you were disappearing.""",

    # ── Essay Grading ─────────────────────────────────────────────────────
    "essay/essay_A.txt": """\
Should Public Transport Be Free?

Public transport is something everyone uses and it should probably be free. There are a lot of reasons why this would be a good idea and I think most people would agree if they thought about it more carefully.

First of all, pollution is a big problem in cities. Cars create a lot of emissions and this is bad for the environment. If public transport was free, more people would use it instead of driving. Some countries have already tried this and it worked pretty well. Making it free just makes sense because people would obviously choose the free option over paying for gas and parking.

Another reason is that it helps poor people. A lot of people can't afford to take the bus or train every day, and this means they can't get to work or school as easily. When you think about it, transportation is basically a necessity like water or electricity, so maybe it should be treated the same way. Some studies have shown that transportation costs are one of the biggest expenses for low income families, which is really unfair.

People always say "but who's going to pay for it?" which is a fair question I guess. But governments already spend tons of money on roads and highways that mostly benefit car drivers. Why not spend that money on public transport instead? Also, there are other ways to raise money like taxes on pollution or parking. The cost of making transit free is actually not as high as people think because fare collection is expensive too.

One concern is that free transit might get too crowded. This is possible but it doesn't mean we shouldn't try it. Other public services like libraries and parks are free and they work fine. The government would just need to make sure there are enough buses and trains to handle more passengers.

Some people might also argue that if something is free, people won't value it. But this argument doesn't really hold up when you look at other free services. Nobody says we should charge admission to public parks because people won't appreciate them otherwise.

In conclusion, free public transport would be good for the environment, good for equality, and honestly just makes life easier for everyone. Yes there would be challenges but the benefits clearly outweigh the costs. Cities should seriously consider making this change because the current system isn't working that well anyway.""",

    "essay/essay_B.txt": """\
Should Public Transport Be Free?

As cities around the world grapple with congestion, pollution, and inequality, the idea of making public transport free has gained traction. While critics raise valid concerns about funding and overcrowding, the benefits of eliminating fares — for the environment, for low-income communities, and for urban mobility as a whole — make a compelling case for this policy shift.

The most immediate argument for free public transport is environmental. Private cars remain one of the largest sources of urban carbon emissions, and getting people out of cars requires making the alternative genuinely attractive. When Luxembourg made all public transport free in 2020, ridership increased significantly, and the country saw measurable reductions in road traffic. Fare-free systems remove one of the key barriers to adoption — cost — and make the choice to leave the car at home easier, especially for short trips where the mental arithmetic of "is this ride worth the fare?" often tips in favor of driving.

Beyond the environment, free transport is a powerful tool for reducing inequality. Transportation costs eat up a disproportionate share of low-income household budgets. In many American cities, a monthly transit pass can cost over $100, which represents a real burden for someone earning minimum wage. Eliminating this cost effectively gives every resident equal access to jobs, healthcare, education, and social opportunities regardless of income. This isn't just a matter of fairness; it's economically productive. Workers who can reliably reach their jobs contribute more to the economy than those who miss shifts because they couldn't afford a bus fare.

The most common objection is funding. Public transport systems are expensive to operate, and fares typically cover only a fraction of operating costs — in most cities, between 20 and 40 percent. However, this means that 60 to 80 percent of the costs are already covered by taxes and subsidies. Closing the remaining gap is not as dramatic as it sounds. Cities could fund free transit through modest increases in parking fees, congestion charges, or reallocating road-construction budgets. The savings from eliminating fare collection infrastructure — ticket machines, enforcement staff, payment processing systems — also offset part of the cost.

There are legitimate concerns about overcrowding and service quality, and these should not be dismissed. A sudden influx of riders without corresponding investment in capacity could degrade the experience for existing users. Any move toward free transit must be paired with investment in expanded service, or the policy will backfire.

Nevertheless, the weight of evidence suggests that free public transport is a policy worth pursuing. It addresses climate change, reduces inequality, and improves urban livability. The funding challenges, while real, are manageable — especially when weighed against the enormous hidden costs of car-dependent infrastructure. Cities willing to make this investment would be building a more equitable and sustainable future for all their residents.""",

    "essay/essay_C.txt": """\
The first time I really failed at something was when I tried out for the school basketball team in eighth grade. I had been playing in the driveway all summer and I thought I was pretty good. My friend Marcus and I would play one-on-one for hours and I could beat him most of the time, so I figured making the team would be no problem.

The tryouts were three days long and I realized pretty quickly that I was out of my league. The other kids were faster and taller and they knew plays and formations that I had never even heard of. I didn't know what a "pick and roll" was. When the coach asked us to run a fast break drill, I had to watch other people do it first and I still got it wrong.

The worst part was the second day when we had to do a scrimmage. I got put on a team with kids I didn't know and I could tell they were annoyed with me because I kept being in the wrong position. One kid said "just stay out of the way" and I pretty much did for the rest of the game. I barely touched the ball.

When the list went up on Friday, my name wasn't on it. Obviously. Marcus made it though, which was hard because I thought we were at the same level. Turns out playing in your driveway and playing on an actual team are totally different things.

I was really upset for a while. I told my mom I didn't care but she knew I did. I thought about trying out again the next year but I didn't. Instead I joined the debate team, which sounds like a weird pivot but it was actually good for me. I found out I was better at arguing than dribbling.

Looking back I think what I learned is that being okay at something in a casual setting doesn't mean you're actually good at it. There's a difference between doing something for fun and doing it competitively. I also learned that failing at one thing can push you toward something else that ends up being a better fit. I never would have tried debate if I had made the basketball team, and debate ended up being one of the best experiences of high school.

I still think about that kid who told me to stay out of the way sometimes. It was mean but he wasn't wrong. I was in over my head and I couldn't see it because I didn't have enough experience to know what good actually looked like. That's probably the biggest lesson — you don't know what you don't know until you put yourself in a situation where it becomes obvious.

So yeah, I failed at basketball tryouts. It felt terrible at the time but it worked out fine. I think everyone needs at least one experience like that to understand that failure isn't the end of anything, it's just information.""",

    "essay/essay_D.txt": """\
The green light at the end of Daisy's dock is probably the most famous symbol in The Great Gatsby, and maybe in all of American literature. Fitzgerald uses it to represent Gatsby's dream of being with Daisy, but it also stands for something bigger — the American Dream itself and how it can destroy you.

The first time we see the green light is at the end of Chapter 1, when Nick sees Gatsby standing alone in the dark reaching toward it. At this point we don't know what it means yet, but it's clearly important to Gatsby. He's reaching for it like it's something he can almost touch, which sets up the whole idea that Gatsby is always reaching for something just out of his grasp. This is a pretty effective image because it shows us what kind of person Gatsby is before we even really know him.

As the novel goes on, the green light becomes connected to Gatsby's obsession with Daisy. He built his whole mansion across the bay from her house just so he could look at that light. Everything he did — the money, the parties, the fake persona — was all about getting closer to what the light represents. But the irony is that the green light represents something that doesn't really exist. The Daisy that Gatsby is in love with is not the real Daisy. She's an idea he created, and the green light is the symbol of that idea.

There's a scene in Chapter 5 where Gatsby and Daisy are reunited and the narrator says something about how the green light is now just a light on a dock again. It lost its meaning because Gatsby is actually with Daisy. But of course this doesn't last because real Daisy can't live up to dream Daisy. The green light was more powerful as a symbol than the actual thing it represented, which is kind of sad when you think about it.

At the end of the novel, Nick talks about the green light one more time and connects it to the American Dream — the idea that we're all "boats against the current, borne back ceaselessly into the past." The green light becomes about all of us, not just Gatsby. We all have something we're reaching for that keeps moving further away the closer we get.

In conclusion, the green light works on multiple levels. It's about Gatsby and Daisy specifically, but also about desire and illusion in general. Fitzgerald is saying that the things we want most are often the things that aren't real, and that pursuing them can end in tragedy. The green light is such a powerful symbol because it's so simple — just a colored light across the water — but it carries so much meaning.""",

    "essay/essay_E.txt": """\
The five-day, forty-hour workweek is not a law of nature. It's a convention, inherited from early 20th-century labor negotiations, and there is growing evidence that it may no longer serve workers, employers, or society particularly well. Companies should seriously consider adopting a four-day workweek — not as a perk, but as a structural improvement that benefits everyone involved.

The most compelling argument for a shorter workweek is productivity. Counterintuitive as it may seem, working fewer hours often results in more output. Microsoft Japan's 2019 trial of a four-day week saw productivity increase by 40%. Iceland ran the world's largest trial between 2015 and 2019, involving over 2,500 workers across diverse sectors, and found that productivity remained the same or improved in the majority of workplaces. These are not outliers. Research consistently shows that after about six hours of focused cognitive work, most people's output declines sharply. The final hours of a long workday are often the least productive, filled with busywork, unnecessary meetings, and diminishing concentration.

Beyond productivity, the mental health benefits are substantial and well-documented. Burnout has reached crisis proportions — the WHO formally recognized it as an occupational phenomenon in 2019, and surveys consistently find that over 70% of workers report experiencing burnout symptoms. A four-day week directly addresses this by providing a genuine third day for recovery, personal obligations, and rest. The Iceland trial found significant reductions in stress and improvements in reported well-being among participants, without any corresponding decline in output.

Critics often argue that certain industries — healthcare, retail, customer service — simply cannot operate on a four-day schedule. This is a legitimate concern but not an insurmountable one. A four-day workweek for individual employees does not mean a four-day operating schedule for the business. Staggered schedules, rotating teams, and flexible arrangements can maintain seven-day coverage while still giving each worker an extra day off. This is already standard practice in industries with shift work; it merely requires thoughtful implementation rather than a blanket policy.

The economic objection — that reduced hours mean reduced output and therefore reduced revenue — is directly contradicted by the evidence. When Microsoft Japan's employees worked four days instead of five, the company also saw a 23% decrease in electricity costs and significant savings in other operational expenses. Fewer days in the office means lower overhead. Combined with maintained or improved productivity, the net financial impact is often positive, not negative.

There is also a broader societal argument. A culture of overwork contributes to inequality: it penalizes caregivers (disproportionately women), reduces civic participation, and diminishes quality of life in ways that GDP figures cannot capture. Giving workers time back is not just an economic calculation; it is an investment in the kind of society we want to build.

The evidence from trials around the world points in the same direction: a four-day workweek is not a utopian fantasy. It is a practical policy change that improves productivity, reduces burnout, and costs less than most people assume. The question is not whether companies can afford to try it. It is whether they can afford not to.""",

    "essay/essay_F.txt": """\
Since the pandemic, a lot of schools and colleges have started offering more online classes, and this has made a lot of people think about which kind of learning is actually better. Both online and in-person learning have their advantages and disadvantages, and honestly I think it depends on the person and the situation.

One of the biggest advantages of online learning is convenience. You can attend class from anywhere, which is great for people who have jobs or family responsibilities. You don't have to commute, and in a lot of cases you can watch lectures at your own pace. This is especially helpful for people who live far from campus or have disabilities that make getting to class difficult. My cousin takes online classes because she works full-time and she says she couldn't go back to school any other way.

On the other hand, in-person learning has some things that online learning just can't replicate. Being in a classroom with other students and a teacher creates a kind of energy that you don't get on a Zoom call. You can ask questions more naturally, have discussions that actually flow, and you're less likely to get distracted because you're physically present. I know from my own experience that I pay way more attention in a physical classroom than I do staring at my laptop at home.

Another difference is social interaction. College is supposed to be about more than just classes — it's about meeting people, making connections, and having experiences outside the classroom. It's hard to make friends over a screen. Study groups, hallway conversations, and even just sitting next to someone in lecture are all things that build a social network. Online learning can feel isolating, especially for younger students who are still developing their social skills.

However, online learning has gotten a lot better in recent years. There are more interactive tools, discussion boards, breakout rooms, and other ways to engage. Some students actually participate more online because they're less shy about typing a question than raising their hand. And for subjects that are mostly lecture-based, there might not be a huge difference between watching in person and watching on screen.

The cost is another factor. Online programs are often cheaper because schools don't need as much physical space. Students save money on housing, transportation, and food. For people who are paying their own way through school, this can make a big difference.

In conclusion, I don't think one is universally better than the other. In-person learning is probably better for most traditional students who want the full college experience and learn best through interaction. Online learning is better for people who need flexibility and can motivate themselves to study independently. The ideal solution might be a hybrid model that combines the best of both, but that's easier said than done. Either way, the most important thing is that the education is good quality, regardless of how it's delivered.""",

    "essay/essay_G.txt": """\
Every time budget cuts come around, someone suggests closing a library. The argument is always the same: everything is online now, who needs a building full of books? This view is not just wrong — it reflects a deep misunderstanding of what libraries actually do in 2024.

Yes, most books can be found digitally. But libraries were never just about books. They are one of the last truly public spaces in American life — places where anyone can walk in without paying, without buying something, without justifying their presence. In a country where nearly every public space has been commercialized, this matters more than people realize.

Consider what a typical library actually provides. Free internet access, which matters enormously for the roughly 25 million Americans without broadband at home. Job search assistance, resume workshops, and computer literacy classes. Children's programs that serve as de facto childcare for working parents. Meeting rooms for community organizations. In many cities, libraries are the primary point of access for government services — passport applications, tax forms, voter registration.

These are not theoretical benefits. Go to any urban library on a weekday and you will find it full. Homeless individuals using the restroom and warming up. Immigrants practicing English. Teenagers doing homework because their apartment is too loud. Elderly people reading newspapers and talking to the librarians who might be the only people they speak to that day. The library serves all of them without judgment and without a cover charge.

The digital argument also ignores equity. Not everyone has a laptop and a quiet room and a good Wi-Fi connection. For millions of Americans, the library IS the internet. Cutting library funding because "everything is online" is like cutting food stamps because restaurants exist — it only makes sense if you forget that access is not universal.

There is also something to be said for libraries as a democratic institution. In a time when algorithms decide what information you see, a library offers something radical: a curated, unfiltered collection of human knowledge that you can browse freely. Nobody is tracking what you read. Nobody is serving you content based on your profile. This kind of intellectual freedom is increasingly rare and increasingly valuable.

Are there things libraries could do better? Of course. Many need updated technology, extended hours, and more funding for programming. But the solution to an underfunded institution is not to defund it further. It's to invest in what's working.

The people who argue that libraries are obsolete are, almost without exception, people who don't need them. That's not an argument for closing them. It's an argument for why they matter.""",

    "essay/essay_H.txt": """\
Last summer I volunteered at the Sunny Days Animal Shelter and it was one of the most meaningful experiences of my life. I learned a lot about animals and also about myself and I think everyone should try volunteering at least once.

When I first got there I was nervous because I didn't know what to expect. The shelter was bigger than I thought and there were so many animals. There were dogs and cats and even some rabbits. The people who worked there were really nice and showed me what to do. My main job was helping clean the cages and feed the animals and sometimes taking the dogs for walks.

The dogs were my favorite part. There was this one dog named Biscuit who was a brown lab mix and he was really shy at first. The staff said he had been abandoned by his owner and didn't trust people very much. I made it my goal to get him to trust me. Every day I would sit by his cage and talk to him quietly and after about a week he started coming up to me and letting me pet him. It was such an amazing feeling when he finally wagged his tail at me.

I also learned that working at a shelter is not all fun and games. It was really sad seeing animals that nobody wanted. Some of them had been there for months and you could tell they were depressed. The older cats especially seemed to have given up hope. It made me realize how important it is to adopt from shelters instead of buying from breeders.

Another thing I learned is that volunteering teaches you responsibility. I had to show up on time and do tasks that weren't always fun like cleaning up messes. But knowing that the animals depended on me made me want to be reliable. My mom said she noticed I became more responsible at home too which was cool.

By the end of the summer Biscuit got adopted by a family with kids and I cried when they took him home. Happy tears though because I knew he was going to a good place. The staff gave me a certificate and said I was one of the best volunteers they had which made me feel proud.

In conclusion volunteering at the animal shelter taught me about compassion, responsibility, and the importance of helping others. I plan to go back next summer and I would encourage anyone who wants to make a difference to try volunteering. You will get more out of it than you expect.""",

    "essay/essay_I.txt": """\
The average American college student sleeps approximately 6.4 hours per night, well below the 7-9 hours recommended by the National Sleep Foundation. This chronic sleep deficit is not merely an inconvenience — it is a public health issue with measurable consequences for academic performance, mental health, and physical well-being.

The cognitive effects of sleep deprivation are well established in the research literature. A landmark study by Williamson and Feyer (2000) found that seventeen hours without sleep produces impairment equivalent to a blood alcohol content of 0.05% — roughly the level at which most people begin to experience reduced coordination and slowed reaction time. For students pulling all-nighters before exams, this finding is particularly troubling: they are attempting complex cognitive tasks in a neurologically compromised state. Research on memory consolidation further underscores the problem. Sleep, particularly slow-wave and REM stages, plays a critical role in transferring information from short-term to long-term memory. Students who sacrifice sleep to study more may paradoxically retain less of what they studied.

The mental health implications are equally concerning. Large-scale surveys consistently show a strong correlation between insufficient sleep and elevated rates of anxiety, depression, and suicidal ideation among college students. A 2019 study in the Journal of American College Health found that students sleeping fewer than six hours per night were 2.5 times more likely to report symptoms of depression than those sleeping eight hours. While correlation does not establish causation, experimental studies in which participants' sleep is deliberately restricted show rapid onset of mood disturbances, suggesting the relationship is at least partially causal.

Physical health also suffers. Sleep deprivation disrupts glucose metabolism, increases cortisol production, and weakens immune function. Students who sleep poorly are more likely to get sick, gain weight, and experience chronic fatigue — all of which create a feedback loop that further impairs academic performance. The irony is self-reinforcing: students sacrifice sleep to keep up with demanding schedules, which makes them less efficient, which creates more pressure, which leads to even less sleep.

What can be done? At the individual level, sleep hygiene education — reducing screen time before bed, maintaining consistent sleep schedules, limiting caffeine after noon — can help. But systemic changes matter more. Universities that have shifted class start times to 9:00 AM or later have seen measurable improvements in both attendance and performance. Some institutions have begun prohibiting exams before 9 AM and discouraging faculty from assigning work due at midnight. These are small structural changes that acknowledge a biological reality: students cannot learn effectively if they are not sleeping.

The evidence is clear. Sleep deprivation among students is widespread, harmful, and addressable. Treating it as a badge of honor or an inevitable feature of college life ignores decades of research showing that it undermines the very purpose of education. If we want students to learn, we need to let them sleep.""",

    "essay/essay_J.txt": """\
Video games are a big topic these days and everyone has an opinion about whether they are good or bad for teenagers. I think video games are good for teenagers and in this essay I will explain why.

First of all video games are fun. After a long day at school teenagers need a way to relax and unwind. Playing games is a way to do that. Its better than just watching TV because your actually doing something interactive. Some people say teenagers should go outside instead but thats not always possible especially in bad weather or if you live somewhere without a lot of space.

Video games also help you make friends. A lot of games are multiplayer now and you can play with people from all over the world. My best friend and I started hanging out because we both played the same game. Studies have shown that gaming can improve social skills which is important for teenagers who might be shy or have trouble making friends in real life.

Another good thing about video games is that they can teach you stuff. There are games about history, science, math, and other subjects. Even games that arent educational can teach problem solving and strategic thinking. Minecraft for example is basically a creativity and engineering game. And games like Civilization teach you about history and geography.

People always bring up violence when they talk about video games. Yes some games are violent but that doesnt mean they make people violent. There have been lots of studies on this and most of them say there is no direct link between video games and real world violence. Millions of people play violent games and dont go out and hurt anyone. Its unfair to blame games for problems that have much deeper causes.

Some people also say games are addictive and teenagers play too much. This might be true in some cases but its not a reason to say games are bad. Lots of things can be overdone — sports, social media, even reading. The answer is moderation not elimination. Parents should set limits but they shouldnt ban games entirely.

In conclusion video games are good for teenagers because they are fun, social, and educational. The arguments against them are mostly based on stereotypes and fear. As long as teenagers play in moderation and balance gaming with other activities there is nothing wrong with enjoying video games.""",

    # ── Hiring ────────────────────────────────────────────────────────────
    "hiring/cv_01.txt": """\
ALEX MORGAN
alex.morgan@email.com | (555) 312-4789 | linkedin.com/in/alexmorgan

SUMMARY
Data analyst with 6 years of experience in reporting, SQL-based data pipelines, and dashboard development. Looking for a mid-level role where I can contribute to data-driven decision-making.

EDUCATION
B.A. in Statistics — University of Wisconsin-Milwaukee, 2018
GPA: 3.4 | Relevant coursework: Regression Analysis, Database Systems, Survey Methods

EXPERIENCE

Data Analyst | Midwest Health Partners, Milwaukee, WI | Jan 2021 - Present
- Built and maintained Tableau dashboards tracking patient intake, wait times, and satisfaction scores across 12 clinic locations
- Wrote SQL queries to consolidate data from three legacy systems into a unified reporting layer, reducing monthly report prep time by ~40%
- Assisted in an A/B test of a new appointment scheduling flow; prepared the analysis deck for leadership
- Respond to ad hoc data requests from clinical and operations teams

Junior Data Analyst | RetailScope Analytics, Chicago, IL | Jun 2019 - Dec 2020
- Supported the analytics team by pulling weekly sales and inventory reports for ~200 retail clients
- Created Excel-based templates for quarterly business reviews
- Helped migrate a set of recurring reports from Excel to Tableau
- Participated in a cross-functional project to standardize product category taxonomy

Data Intern | City of Milwaukee, Office of Budget & Management | May 2018 - May 2019
- Cleaned and organized datasets for the city's open data portal
- Assisted with visualizations for the annual budget presentation
- Learned to use Python (pandas) for basic data cleaning tasks

SKILLS
- SQL (advanced), Python (pandas, basic scripting), Tableau, Excel (pivot tables, VLOOKUP, macros)
- Familiar with: Git, Jupyter Notebooks, basic R
- Comfortable presenting findings to non-technical stakeholders

CERTIFICATIONS
- Tableau Desktop Specialist (2021)""",

    "hiring/cv_02.txt": """\
PRIYA SHARMA
priya.sharma@email.com | (415) 709-2331 | linkedin.com/in/priyasharma | github.com/psharma-data

SUMMARY
Senior data analyst with 5 years of experience in product analytics, experimentation, and data infrastructure at scale. Skilled in SQL, Python, and statistical modeling. Looking for a mid-level or senior role on a data team focused on product-driven decision-making.

EDUCATION
M.S. in Statistics — Stanford University, 2021
B.S. in Mathematics — UC Berkeley, 2019 (Magna Cum Laude)

EXPERIENCE

Data Analyst II | Google, Mountain View, CA | Aug 2021 - Present
- Designed and analyzed A/B tests for Google Maps features, supporting product decisions affecting 1B+ monthly users
- Built automated anomaly detection pipeline in Python (pandas, scipy) that reduced time-to-detection for metric regressions by 60%
- Developed Looker dashboards used by 3 product teams for weekly business reviews; standardized KPI definitions across teams
- Led a cross-functional analysis of user retention drivers that informed the H2 2023 product roadmap
- Mentored 2 junior analysts on SQL best practices and experiment design

Data Science Intern | Stripe, San Francisco, CA | Jun 2020 - Sep 2020
- Built a churn prediction model (logistic regression + gradient boosting) for SMB merchants; model deployed to production with 0.82 AUC
- Automated weekly fraud reporting pipeline, eliminating 5 hours of manual work per analyst per week
- Presented findings to the Risk team leadership; recommendations adopted for Q4 fraud policy updates

Research Assistant | Stanford Statistics Department | Sep 2019 - Jun 2021
- Assisted with research on causal inference methods for observational studies
- Co-authored a workshop paper on doubly-robust estimators (presented at JSM 2021)

SKILLS
- SQL (expert), Python (pandas, numpy, scipy, scikit-learn, matplotlib), R
- Tools: Looker, Tableau, BigQuery, Jupyter, Git, Airflow
- Methods: A/B testing, causal inference, regression modeling, time-series analysis, survival analysis
- Strong communicator; experienced presenting to both technical and non-technical audiences

CERTIFICATIONS
- Google Advanced Data Analytics Certificate (2022)""",

    "hiring/cv_03.txt": """\
JORDAN KIMBALL
jordan.kimball@email.com | (312) 558-0142 | linkedin.com/in/jordankimball

SUMMARY
Software engineer with 4 years of backend development experience, transitioning into data analytics. Strong SQL and Python skills from building data pipelines and internal tools. Looking for a mid-level data analyst role where I can apply my engineering background to analytics problems.

EDUCATION
B.S. in Computer Science — DePaul University, Chicago, IL, 2020
GPA: 3.2

EXPERIENCE

Software Engineer | DataBridge Solutions, Chicago, IL | Jul 2020 - Present
- Developed and maintained ETL pipelines processing 2M+ records daily using Python and Apache Airflow
- Built REST APIs for internal reporting tools used by the operations team
- Wrote complex SQL queries against PostgreSQL databases for ad hoc data requests from product managers
- Created a monitoring dashboard (Grafana) tracking pipeline health metrics and data quality scores
- Participated in migration from on-prem SQL Server to cloud-based Snowflake data warehouse

Software Engineering Intern | LoopNet Technologies, Chicago, IL | May 2019 - Aug 2019
- Built automated test suites for the QA team using Python and Selenium
- Assisted in database schema design for a new customer-facing feature
- Participated in daily standups and sprint planning (Agile/Scrum)

Freelance Projects | 2018 - 2020
- Built a personal finance tracking app using Flask and SQLite
- Created web scrapers for small business clients to aggregate competitor pricing data

SKILLS
- Python (advanced), SQL (PostgreSQL, Snowflake), JavaScript (basic)
- Tools: Airflow, Grafana, Git, Docker, VS Code
- Familiar with: Tableau (self-taught), pandas, basic statistics
- Limited experience with machine learning and statistical modeling

CERTIFICATIONS
- AWS Cloud Practitioner (2022)
- Google Data Analytics Professional Certificate (in progress)""",

    "hiring/cv_04.txt": """\
RACHEL TORRES
rachel.torres@email.com | (646) 887-4512 | linkedin.com/in/racheltorres

SUMMARY
Marketing analyst with 5 years of experience in campaign performance tracking, customer segmentation, and reporting. Seeking a mid-level data analyst role to broaden my analytics career beyond marketing.

EDUCATION
B.A. in Economics — SUNY Binghamton, 2019
Minor in Mathematics | Relevant coursework: Econometrics, Statistics, Data Visualization

EXPERIENCE

Senior Marketing Analyst | BrightPath Digital, New York, NY | Mar 2022 - Present
- Manage end-to-end analytics for paid media campaigns across Google, Meta, and TikTok ($3M+ annual spend)
- Built customer segmentation model using k-means clustering in Python; improved email campaign targeting by 28%
- Created Tableau dashboards tracking CAC, LTV, and ROAS across 4 product lines; used by C-suite in quarterly reviews
- Designed attribution modeling framework to measure incremental impact of campaigns across channels
- Write SQL queries against Snowflake to extract and transform marketing data for analysis

Marketing Analyst | Greenleaf Consumer Brands, New York, NY | Jun 2019 - Feb 2022
- Tracked and reported on campaign KPIs (CTR, conversion rate, CPA) across email, social, and search channels
- Built Excel models for budget allocation across campaigns; migrated these to Tableau in 2021
- Supported the data team in building a unified customer data platform (Segment)
- Conducted post-campaign analyses and presented recommendations to marketing leadership

Marketing Intern | Ogilvy, New York, NY | Jan 2019 - May 2019
- Assisted with competitive analysis and market research for CPG clients
- Compiled weekly media monitoring reports

SKILLS
- SQL (Snowflake, BigQuery), Python (pandas, scikit-learn), Excel (advanced), Tableau
- Marketing-specific: Google Analytics, Meta Ads Manager, Segment, Mixpanel
- Methods: Segmentation, attribution modeling, A/B testing, regression analysis
- Experienced presenting insights to non-technical stakeholders

CERTIFICATIONS
- Google Analytics Certification (2021)
- HubSpot Inbound Marketing (2020)""",

    "hiring/cv_05.txt": """\
TYLER CHEN
tyler.chen@email.com | (503) 221-6788 | linkedin.com/in/tylerchen-ux

SUMMARY
UX researcher with 4 years of experience in mixed-methods research and data analysis. Looking to transition into a data analyst role, leveraging my quantitative research skills and experience with large-scale user data.

EDUCATION
M.A. in Human-Computer Interaction — University of Washington, 2020
B.A. in Psychology — University of Oregon, 2018

EXPERIENCE

UX Researcher | CloudBase Software, Portland, OR | Sep 2020 - Present
- Designed and analyzed surveys (n=2,000+) using Qualtrics; performed statistical analysis in R (t-tests, ANOVA, regression)
- Conducted A/B test analysis for product features; built reporting templates in Google Sheets and Looker
- Queried user event data in BigQuery to support behavioral analyses and funnel optimization
- Created data visualizations for product team presentations; transitioned team reporting from slides to Looker dashboards
- Led a usability benchmarking study that informed the redesign of the onboarding flow (15% improvement in activation rate)

UX Research Intern | Intel, Hillsboro, OR | Jun 2019 - Aug 2019
- Conducted user interviews and card-sorting exercises for an internal tool redesign
- Analyzed survey data using SPSS; created summary reports for the design team

Research Assistant | UW Interaction Lab | Sep 2018 - Jun 2020
- Assisted with experimental design and data collection for studies on mobile accessibility
- Cleaned and analyzed experimental data using R and Python

SKILLS
- R (advanced), SQL (BigQuery, intermediate), Python (pandas, basic), Excel/Sheets
- Tools: Looker, Qualtrics, SPSS, Figma, Google Analytics
- Methods: Survey design, A/B testing, statistical analysis (hypothesis testing, regression), qualitative coding
- Strong background in experimental design and research methodology

CERTIFICATIONS
- Google UX Design Certificate (2021)""",

    "hiring/cv_06.txt": """\
MARIA DELGADO
maria.delgado@email.com | (210) 334-9021 | linkedin.com/in/mariadelgado

SUMMARY
Former high school math teacher transitioning into data analytics. Completed a data analytics bootcamp and have been building projects and freelancing for the past year. Strong quantitative foundation and experience explaining complex concepts to diverse audiences.

EDUCATION
Data Analytics Bootcamp — General Assembly, Remote, 2024 (12 weeks)
Coursework: SQL, Python, Tableau, statistics, machine learning fundamentals

M.Ed. in Mathematics Education — University of Texas at San Antonio, 2017
B.S. in Mathematics — Texas State University, 2015

EXPERIENCE

Freelance Data Analyst | Self-Employed | Jan 2024 - Present
- Built Tableau dashboards for 3 small business clients tracking sales, inventory, and customer metrics
- Performed customer churn analysis for a local SaaS company using Python (pandas, scikit-learn); identified 4 key retention drivers
- Wrote SQL queries to clean and consolidate client data from multiple sources (CSV, Google Sheets, MySQL)

Math Teacher | Roosevelt High School, San Antonio, TX | Aug 2017 - Jun 2023
- Taught Algebra II, Pre-Calculus, and AP Statistics to classes of 25-35 students
- Developed data-driven approach to tracking student performance; created spreadsheet dashboards monitoring grades, attendance, and test score trends
- Served as department data lead, analyzing standardized test results to identify areas for curriculum improvement
- Named Teacher of the Year (2021); consistently rated in top 10% of campus evaluations

Math Tutor | Mathnasium, San Antonio, TX | 2015 - 2017
- Tutored students grades 6-12 in algebra, geometry, and statistics

SKILLS
- SQL (MySQL, intermediate), Python (pandas, matplotlib, basic scikit-learn), Tableau, Excel (advanced)
- Statistics: hypothesis testing, regression, ANOVA, descriptive statistics
- Strong communication and presentation skills from 6 years of teaching
- Familiar with: Git, Jupyter Notebooks, Google BigQuery

PROJECTS
- School District Analysis (portfolio project): Analyzed public Texas education data to identify factors correlated with student achievement; presented findings in a Tableau story
- Austin Housing Market Dashboard: Built interactive Tableau dashboard using Zillow API data""",

    "hiring/cv_07.txt": """\
SARAH WELLINGTON
sarah.wellington@email.com | (617) 445-2890 | linkedin.com/in/sarahwellington

SUMMARY
Data analyst returning to the workforce after a four-year career break. Previously worked in business intelligence at two financial services firms. Skilled in SQL, Tableau, and Excel. Currently updating my skills through coursework and personal projects.

EDUCATION
B.S. in Information Systems — Northeastern University, Boston, MA, 2014
GPA: 3.5 | Co-op: State Street Corporation (6 months)

EXPERIENCE

Career Break | 2020 - 2024
- Completed online courses in Python for data analysis (Coursera), advanced SQL (DataCamp), and statistics refresher (Khan Academy)
- Built 3 portfolio projects (see Projects section)

Business Intelligence Analyst | Fidelity Investments, Boston, MA | Mar 2018 - Feb 2020
- Built and maintained Tableau dashboards for the retail brokerage division, tracking AUM, trade volumes, and client acquisition
- Wrote SQL queries (Teradata) to extract and transform data for weekly and monthly reporting
- Automated several recurring Excel reports using VBA, saving approximately 8 hours per week
- Collaborated with product managers to define KPIs for a new mobile trading feature

Data Analyst | BankNorth Analytics, Boston, MA | Jul 2014 - Feb 2018
- Supported the risk analytics team with data pulls, report generation, and quality assurance
- Created Excel-based dashboards tracking loan delinquency rates across consumer lending portfolios
- Learned SQL on the job; progressed from basic queries to writing complex joins and window functions
- Participated in a data warehouse migration project (Oracle to Teradata)

SKILLS
- SQL (Teradata, PostgreSQL — advanced), Excel (VBA, pivot tables, advanced formulas), Tableau
- Python (pandas, matplotlib — intermediate, recently self-taught)
- Familiar with: Git, Jupyter Notebooks, Power BI
- Domain knowledge: financial services, business intelligence, regulatory reporting

PROJECTS (2024)
- Boston Housing Price Analysis: Regression model in Python predicting housing prices; published on GitHub
- Investment Portfolio Tracker: Tableau dashboard with automated data refresh from public API
- Customer Segmentation: K-means clustering project using retail transaction data (Kaggle dataset)

CERTIFICATIONS
- Tableau Desktop Specialist (2019, currently renewing)""",

    "hiring/cv_08.txt": """\
DYLAN PARK
dylan.park@email.com | (773) 901-2244 | linkedin.com/in/dylanpark

SUMMARY
Recent data analytics bootcamp graduate looking for my first role in data. Eager to learn and grow. Interested in using data to make a difference.

EDUCATION
Data Analytics Bootcamp — Thinkful, Online, 2024 (6 months)
Coursework: Excel, SQL basics, Tableau, intro to Python

B.A. in Communications — Loyola University Chicago, 2022

EXPERIENCE

Barista / Shift Lead | Blue Bottle Coffee, Chicago, IL | Aug 2022 - Present
- Managed daily operations and inventory for a high-volume location
- Tracked sales data in spreadsheets and reported weekly numbers to store manager
- Trained 5 new hires on POS system and store procedures

Intern | Chicago Tribune (Marketing Department) | Jan 2022 - May 2022
- Compiled social media metrics into weekly reports for the marketing team
- Used Google Sheets to track article performance and engagement rates
- Assisted with audience survey distribution and basic data entry

Student Worker | Loyola University Library | Sep 2020 - Dec 2021
- Helped students with research databases and citation software
- Maintained records and filing systems

SKILLS
- Excel (intermediate), SQL (basic SELECT, JOIN, WHERE), Tableau (basic), Python (very basic)
- Familiar with: Google Analytics, Google Sheets, PowerPoint
- Strong communication and customer service skills
- Quick learner, detail-oriented, team player

PROJECTS (Bootcamp)
- Sales Dashboard: Created a Tableau dashboard using sample retail data
- Movie Database: Wrote SQL queries against an IMDB dataset to answer business questions
- Survey Analysis: Used Python to analyze a class survey and create visualizations

CERTIFICATIONS
- Google Data Analytics Professional Certificate (in progress)""",

    "hiring/cv_09.txt": """\
DR. JAMES OKAFOR
james.okafor@email.com | (734) 228-5491 | linkedin.com/in/jamesokafor | scholar.google.com/jamesokafor

SUMMARY
Computational scientist with a PhD in Biostatistics and 3 years of postdoctoral research experience. Seeking to transition from academia to industry. Extensive experience in statistical modeling, data analysis, and scientific programming.

EDUCATION
Ph.D. in Biostatistics — University of Michigan, Ann Arbor, 2021
Dissertation: "Bayesian Methods for High-Dimensional Survival Analysis in Oncology Data"

M.S. in Statistics — University of Michigan, Ann Arbor, 2018
B.S. in Mathematics — University of Lagos, Nigeria, 2016 (First Class Honours)

EXPERIENCE

Postdoctoral Research Fellow | Michigan Medicine, Ann Arbor, MI | Sep 2021 - Present
- Developed Bayesian hierarchical models for multi-site clinical trial data (R, Stan)
- Analyzed large-scale EHR datasets (n > 500,000) using R and SQL to identify predictors of treatment response
- Published 6 peer-reviewed papers in journals including Biostatistics, Statistics in Medicine, and JAMA Oncology
- Taught graduate-level courses in survival analysis and Bayesian inference
- Supervised 3 master's students on their thesis projects

Graduate Research Assistant | University of Michigan, Dept. of Biostatistics | 2016 - 2021
- Developed novel statistical methodology for high-dimensional variable selection in survival models
- Programmed simulation studies in R and C++ to evaluate estimator performance
- Collaborated with clinicians at Michigan Medicine on 4 applied research projects
- Won the department's Outstanding Student Paper Award (2020)

SKILLS
- R (expert: tidyverse, ggplot2, Stan, survival, glmnet), Python (intermediate: pandas, numpy, scipy)
- SQL (intermediate), Stata, SAS, MATLAB, C++ (basic)
- Methods: Bayesian inference, survival analysis, causal inference, mixed models, high-dimensional statistics, clinical trial design
- Tools: Git, LaTeX, RMarkdown, Jupyter, Linux/HPC cluster computing
- 6 first-author publications, 4 co-authored; h-index: 8

CERTIFICATIONS
- None (extensive peer-reviewed publication record)""",

    "hiring/cv_10.txt": """\
KEVIN MARSH
kevin.marsh@email.com | (614) 332-7710 | linkedin.com/in/kevinmarsh

SUMMARY
Looking for a data analyst position. I have experience with Excel and some SQL. Hard worker and fast learner.

EDUCATION
B.S. in Business Administration — Ohio University, Athens, OH, 2020
Concentration: General Management | GPA: 2.8

EXPERIENCE

Administrative Assistant | Midwest Insurance Group, Columbus, OH | Mar 2021 - Present
- Enter policyholder information into company database system
- Generate monthly reports using Excel templates provided by supervisor
- Respond to data requests from agents by pulling information from the CRM
- Maintain and organize digital filing system for claims documents

Sales Associate | Best Buy, Columbus, OH | Jun 2020 - Feb 2021
- Assisted customers with product selection and purchases
- Met monthly sales targets consistently
- Processed returns and exchanges using point-of-sale system

Intern | Ohio University Athletics Department | Sep 2019 - May 2020
- Helped organize and update player statistics database in Excel
- Created simple charts for coaches to review game performance
- Assisted with scheduling and event coordination

SKILLS
- Excel (pivot tables, VLOOKUP, basic formulas)
- SQL (learning — completed a Udemy course)
- Familiar with: Tableau (just started), PowerPoint, Word
- Good communication skills, organized, reliable
- Typing speed: 65 WPM

CERTIFICATIONS
- None currently (plan to start Google Data Analytics certificate)""",

    # ── Medical Triage ────────────────────────────────────────────────────
    "medical/med_01.txt": """\
Persistent headache for about a week. Started mild, has been slowly getting worse. Some sensitivity to light. No fever, no vision changes, no nausea. I've had migraines before but this feels a bit different. Otherwise feeling okay — eating, sleeping, working. Took ibuprofen a couple of times with partial relief.""",

    "medical/med_02.txt": """\
Been having this tightness in my chest on and off for about four days. It's not sharp pain, more like a pressure or heaviness, mostly in the center. It comes and goes — worst when I'm stressed or after climbing stairs, but sometimes it happens when I'm just sitting. No shortness of breath really, no arm pain, no sweating. I'm 38, don't smoke, not overweight. My dad had a heart attack at 62 but he was a smoker. I figured it's probably muscle strain since I started going to the gym again last week after a long break. But it's been bugging me enough that I'm second-guessing myself. Took some antacid which maybe helped a little, hard to tell.""",

    "medical/med_03.txt": """\
I've been having stomach issues for about two weeks now. Started with some bloating and discomfort after eating, which I thought was something I ate. But then I started getting these cramps, mostly in the lower left side, that come in waves. I've also noticed some changes in my bowel habits — alternating between loose stools and constipation, which isn't normal for me. No blood that I've noticed. Lost maybe 3 pounds but I haven't been eating as much because eating makes the cramps worse. No fever, no vomiting. I'm 45 and generally healthy. My mom had colon polyps removed in her 50s. I've been pretty stressed at work lately so I'm wondering if it's just IBS or something.""",

    "medical/med_04.txt": """\
I've got this rash on both forearms that showed up about three days ago. Small red bumps, slightly itchy but not terrible. Doesn't hurt. I think it might be from switching laundry detergent — my partner bought a new brand last week. No blisters, no oozing, no swelling. Not spreading to other parts of my body. No fever or anything else going on. I tried some over-the-counter hydrocortisone cream and it seems to help a bit with the itch. I have no history of skin conditions or allergies, though I've always had somewhat sensitive skin.""",

    "medical/med_05.txt": """\
For the last month or so I've been feeling really tired all the time, even when I sleep a full eight hours. I wake up and I'm still exhausted. My motivation is basically gone — I used to enjoy cooking and going for walks but now I just can't be bothered. I've been kind of foggy mentally, like I can't concentrate at work and I keep forgetting things. My appetite has decreased, I've lost maybe 5 pounds without trying. I'm not having thoughts of hurting myself or anything like that, but I just feel flat, like everything is in grayscale. I'm 34, no major health issues. I did go through a breakup about two months ago but I thought I was handling it okay. Haven't been to a doctor in a while.""",

    "medical/med_06.txt": """\
My 4-year-old has had a fever for three days now. It started at about 101\u00b0F and went up to 103.5\u00b0F yesterday. Children's Tylenol brings it down for a few hours but it keeps coming back. She's been really lethargic — normally she's running around nonstop but she's just been lying on the couch, doesn't want to play, barely watching her shows. She's eating very little, drinking some but not as much as usual. She complained that her neck hurts when I tried to get her to look up at something. No rash that I can see. No cough or runny nose. She did vomit once yesterday morning but nothing since. She goes to preschool and there's been some kind of bug going around.""",

    "medical/med_07.txt": """\
About an hour ago my face started feeling weird on the right side. When I looked in the mirror, the right side of my mouth is drooping — I can't smile evenly. My right arm also feels weak and heavy, like I'm trying to lift it through water. My speech is okay I think but my partner says I'm slurring a little. No headache, no loss of consciousness. I'm 58, I have high blood pressure which I take medication for, and I'm slightly overweight. This came on very suddenly while I was watching TV. I feel confused about whether this is serious or will just pass.""",

    "medical/med_08.txt": """\
I've had a runny nose and sneezing for about five days. Started right when the pollen count went up this week. Eyes are a little itchy and watery. Mild sore throat in the morning that goes away by noon — I think it's from post-nasal drip. No fever, no body aches, no cough really. Energy level is fine, I've been going to work and exercising normally. This happens to me every spring, honestly. Last year I took Claritin and it helped, but I ran out and haven't gotten more yet. No other symptoms.""",

    "medical/med_09.txt": """\
I'm having really bad chest pain that started about 30 minutes ago. It's a crushing, squeezing feeling right in the center of my chest. It's radiating into my left arm and up into my jaw. I'm also short of breath and sweating a lot even though I'm just sitting down. I feel nauseous. I'm 55, male, I have high cholesterol and I used to smoke — quit three years ago. My father died of a heart attack at 60. I took an aspirin because I've heard that's what you're supposed to do. The pain is not going away. My wife wants to call 911 but I wanted to check first how urgent this actually is.""",

    "medical/med_10.txt": """\
I started a new medication about ten days ago — my doctor put me on sertraline (Zoloft) 50mg for anxiety. Since then I've been having some weird symptoms. Pretty nauseous for the first week, which I was told to expect, but now I'm also getting these dizzy spells when I stand up too fast, and my hands have been trembling slightly. I've also been sleeping a lot more than usual, like 10-11 hours and still feeling groggy. The anxiety itself feels a bit better actually. I called the doctor's office and the nurse said to give it two more weeks, but the dizziness is making me nervous. I'm 29, no other medications, no other health issues. I'm eating and drinking normally.""",
}

if __name__ == "__main__":
    for rel_path, content in STIMULI.items():
        full_path = BASE / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        print(f"  wrote {rel_path}")
    print(f"\nDone: {len(STIMULI)} stimulus files created.")
