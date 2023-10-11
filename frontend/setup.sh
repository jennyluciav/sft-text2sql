aws s3 cp schema.png s3://$(python3 -c "import sagemaker; sess = sagemaker.Session(); print(sess.default_bucket())")/schema.png
eb init --region us-east-1 --platform docker text2sql 
eb use text2sql-env
eb deploy text2sql-env
