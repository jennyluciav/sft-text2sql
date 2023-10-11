aws s3 cp schema.png s3://natural-language-to-sql-bucket-$(aws sts get-caller-identity | python3 -c "import sys, json; print(json.load(sys.stdin)['Account'])")/schema.png
eb deploy --region "us-east-1"
