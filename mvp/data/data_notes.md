# ShelfSense Phase 2 Data Notes

This Phase 2 dataset is still synthetic for reproducibility, but it is calibrated to match the ShelfSense proposal more closely than a generic random simulator.

## Fidelity upgrades

- Store-specific Arizona locations: Tempe, Phoenix, Mesa, and Scottsdale.
- Store-specific local event calendar instead of one shared random event stream.
- Arizona-like temperature curve with city offsets and heatwave flags.
- Grocery-relevant holidays and long-weekend periods.
- Promotions that are more likely around weekends, paydays, holidays, and event weekends.
- Different store profiles to reflect campus, urban, suburban, and premium patterns.

## Public-signal inspiration

The calendar is designed to mirror the proposal idea of combining retail demand with outside signals such as holidays and local events. To keep the Phase 2 package reproducible without API keys, the event schedule is a deterministic proxy inspired by the kinds of signals the team proposed to pull from sources such as Nager.Date and Ticketmaster in later phases.

## City event families included

- Tempe: asu_move_in (campus move-in and back-to-school surge), tempe_festival_of_the_arts (downtown arts festival foot traffic), tempe_marathon (race weekend convenience demand), asu_homecoming (football and alumni traffic)
- Phoenix: state_fair (fairgrounds traffic and concession demand), downtown_concert_series (arena and concert district traffic), sports_game (large game-day crowd effects), holiday_market (seasonal downtown shopping crowds)
- Mesa: spring_training (spring baseball tourism and snack demand), mesa_fall_festival (community festival weekend), holiday_lights (family event traffic), swap_meet_weekend (weekend family shopping traffic)
- Scottsdale: golf_tournament (destination sports traffic), art_walk (gallery district tourism), western_week (parade and rodeo foot traffic), holiday_shopping_weekend (high-income seasonal shopping lift)