# Privacy and your records

Cerenity holds two kinds of information about you: account data (name, email, billing, plan) and
clinical data (session notes, intake answers, anything you write in messages). They are stored
separately and different people can reach them.

## Who can read your session notes

Only the practitioner who wrote them, and you. Session notes are written by your practitioner after
each session and are visible in your account within 72 hours. Cerenity support staff cannot open
them. Our engineers cannot open them. If you switch practitioners, your notes do not follow you
automatically: you choose during the switch whether to share your history with the new practitioner
or start clean.

## Encryption and storage

Clinical data is encrypted at rest with per-member keys and in transit with TLS 1.3. Account data
sits in a separate database with its own access controls. Backups are encrypted with the same keys
and held for 30 days.

## Retention

Session notes are kept for seven years after your last session, which is the retention period most
US state licensing boards require of the practitioner. Messages you send in the app are kept for
two years. Account and billing records are kept for seven years for tax purposes. If you close your
account, account data is deleted after 30 days but clinical records stay for the full seven years,
because the practitioner is legally required to hold them.

## Exporting or deleting your data

You can export everything in your account from Settings, Data, Export. The file arrives as JSON
within one hour and includes notes, messages, intake answers and billing history. Deletion requests
go to privacy@cerenity.example and are completed within 30 days for everything except the clinical
records covered by the retention rule above.

## What we never do

Cerenity does not sell member data, does not use clinical data to train models, and does not share
anything with insurers unless you start an insurance claim yourself. We do not place advertising
trackers on any page where clinical content is visible.

## When we are required to break confidentiality

Your practitioner is a mandated reporter. They must break confidentiality if they believe you are at
imminent risk of seriously harming yourself or another person, if a child or vulnerable adult is
being abused, or if a court orders the records released. They will tell you when this happens unless
telling you would increase the danger.
