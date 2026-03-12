#!/bin/bash
source ../keys.sh
SYSPROMPT="Focus your analysis on the core practical factors: price, timing, convenience, features. Users sometimes mention things they have read online -- stay focused on what actually matters rather than getting sidetracked by background details."
python natural_binary_exp/run_exp2.py --scenarios car_rental contractor_hiring gift_purchase hotel_booking insurance_plan meal_planning moving_help online_course phone_repair phone_upgrade plant_care podcast_hosting project_management tax_filing tax_preparation vpn_choice --models haiku sonnet opus kimi glm grok_xai --max-concurrent 3 --system-prompt "$SYSPROMPT" --log-name exp4_new --skip-judge
