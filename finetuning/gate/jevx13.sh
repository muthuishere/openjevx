#!/usr/bin/env bash
# The 13 fundamentals through the real jevx CLI with default thresholds. Usage: jevx13.sh PROFILE
# (No quotes inside a -score spec: jevx keeps them, so SEV="...|high" made the label 'high"'.)
p=$1; n=0; ok=0; uns=0
chk() { # expected, command output
  n=$((n+1)); v=$(echo "$2" | awk '{print $(NF-1)}'); [ "$v" = "$1" ] && ok=$((ok+1)); [ "$v" = "unsure" ] && uns=$((uns+1))
  printf "  %-58s exp=%-10s got=%s\n" "$3" "$1" "$2"; }
is() { echo "$1" | jevx is "$2" --profile $p 2>&1 | tail -1; }
chk yes "$(is '{"stock_units":0}' 'Is the item out of stock?')" "stock 0 -> out of stock"
chk no  "$(is '{"stock_units":500}' 'Is the item out of stock?')" "stock 500 -> out of stock"
chk no  "$(is '{"age":17}' 'Is this person an adult (18 or older)?')" "age 17 -> adult"
chk yes "$(is '{"age":30}' 'Is this person an adult (18 or older)?')" "age 30 -> adult"
chk yes "$(is '{"checkout":"down for all users"}' 'Should on-call be paged?')" "checkout down for all users -> page"
chk no  "$(is '{"task":"making tea","pending":"nothing"}' 'Should on-call be paged?')" "making tea -> page"
chk yes "$(is '{"command":"git push --force origin main"}' 'Does this command destroy data or rewrite shared history?')" "git push --force main -> destructive"
chk no  "$(is '{"command":"ls -la"}' 'Does this command destroy data or rewrite shared history?')" "ls -la -> destructive"
chk no  "$(is '{"log":"INFO GET /health 200 OK"}' 'Is this a failure an on-call engineer should act on?')" "INFO /health 200 -> act"
chk yes "$(is '{"log":"ERROR payment-service: connection refused to db (500)"}' 'Is this a failure an on-call engineer should act on?')" "ERROR db refused 500 -> act"
chk outage  "$(echo '{"pending":["production checkout outage affecting all users","drink tea","reply to newsletter"]}' | jevx pick 'What should an on-call engineer do first?' outage='handle the production outage' tea='drink tea' newsletter='reply to the newsletter' --profile $p 2>&1 | tail -1)" "first: outage/tea/newsletter"
chk billing "$(echo '{"ticket":"I was charged twice this month"}' | jevx pick 'Which team should handle this ticket?' billing='payments and charges' engineering='bugs and outages' sales='new purchases' --profile $p 2>&1 | tail -1)" "team for charged twice"
chk high "$(echo '{"incident":"site completely down for every user"}' | jevx ask -score 'SEV=How severe is this incident?|low;medium;high' --profile $p 2>&1 | tail -1 | awk '{print $2, $3}')" "severity: site down for everyone"
echo "  => $p: correct $ok/$n, unsure $uns"
