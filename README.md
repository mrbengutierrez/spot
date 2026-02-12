Team,

Update on the “getting blocked” question.

I ran maars locally for ~10 hours with the availability agent using the AWS Bedrock Browser against Atrius. 45/60 runs succeeded, and none of the failures looked like provider blocking (more like normal run-to-run issues). This is still a narrow datapoint: one provider, one IP, and one point in time.

I’m not sure what scale we intend to run, so next I want to broaden the test:

* Try multiple providers
* Add concurrency (next test: 4 concurrent sessions from the same IP)

My hunch is that once we scale and run concurrent requests, we may start to see rate limits / blocks. To mitigate that, I’d like to add a recovery mechanism that stays within AWS:

Proposal: start with a small pool of proxy egress IPs (5 to begin)

* Create 5 Elastic IPs and attach each to a small EC2 proxy instance (1 IP per proxy box).
* Maintain a simple table (e.g., DynamoDB) of proxy endpoints with health/status.
* Use “sticky” IP per session/provider by default; rotate only on clear block signals (e.g., repeated 403/429/captcha pages), with backoff and jitter.
* If an IP is “bad,” we can swap it out by allocating a new EIP, associating it to the proxy instance, and releasing the old EIP.

Cost notes (ballpark)

* AWS charges $0.005 per public IPv4 per hour → 5 IPs is ~$0.025/hr, about ~$18/month just for the IPs (assuming 24/7). Compute + egress will be additional.

This isn’t the best possible “rotating proxy” solution overall (third-party services can do this better), but it keeps us in AWS and gives us a straightforward recovery path if we see blocking at higher concurrency.

Next steps I’m planning unless someone objects:

1. Run the 4-concurrent-sessions test on a single IP.
2. If we see blocking/rate limiting, stand up the 5-IP proxy pool and add basic rotate-on-failure logic + metrics (provider/IP/concurrency outcomes).

— [Your Name]
