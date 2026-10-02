package modelsrc

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// s3Store reads one bucket with the AWS default credential chain (env, profile, instance role, IRSA,
// ECS task role), at AWS_ENDPOINT_URL_S3 / AWS_ENDPOINT_URL when set, path-style when asked (Options).
// It only calls GetObject and HeadObject (both covered by s3:GetObject). The region is
// AWS_REGION / the profile's when set; otherwise it starts at us-east-1 and follows the bucket region S3
// names in its x-amz-bucket-region response header, which needs no extra permission.
type s3Store struct {
	bucket string
	cfg    aws.Config
	opt    Options
	mu     sync.Mutex
	client *s3.Client
}

func openS3(ctx context.Context, bucket string, opt Options) (Store, error) {
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("s3://%s: AWS config: %w", bucket, err)
	}
	if cfg.Region == "" {
		cfg.Region = "us-east-1"
	}
	switch v := strings.ToLower(strings.TrimSpace(os.Getenv("AWS_S3_USE_PATH_STYLE"))); v {
	case "", "false", "0":
	case "true", "1":
		opt.PathStyle = true
	default:
		return nil, fmt.Errorf("AWS_S3_USE_PATH_STYLE=%q: want true or false", v)
	}
	s := &s3Store{bucket: bucket, cfg: cfg, opt: opt}
	s.client = s.newClient(cfg.Region)
	return s, nil
}

func (s *s3Store) newClient(region string) *s3.Client {
	return s3.NewFromConfig(s.cfg, func(o *s3.Options) {
		o.Region = region
		o.DisableLogOutputChecksumValidationSkipped = true
		o.UsePathStyle = s.opt.PathStyle
	})
}

// call runs fn, and once more in the bucket's own region if S3 says the bucket lives elsewhere.
func (s *s3Store) call(fn func(*s3.Client) error) error {
	s.mu.Lock()
	c := s.client
	s.mu.Unlock()
	err := fn(c)
	var re *awshttp.ResponseError
	if err != nil && errors.As(err, &re) && re.Response != nil {
		if region := re.Response.Header.Get("X-Amz-Bucket-Region"); region != "" && region != c.Options().Region {
			c = s.newClient(region)
			s.mu.Lock()
			s.client = c
			s.mu.Unlock()
			err = fn(c)
		}
	}
	return s.explain(err)
}

func (s *s3Store) Stat(ctx context.Context, key string) (string, error) {
	var etag string
	err := s.call(func(c *s3.Client) error {
		out, err := c.HeadObject(ctx, &s3.HeadObjectInput{Bucket: &s.bucket, Key: &key})
		if err == nil {
			etag = aws.ToString(out.ETag)
		}
		return err
	})
	return etag, err
}

func (s *s3Store) Fetch(ctx context.Context, key string, w io.Writer) (string, error) {
	var etag string
	err := s.call(func(c *s3.Client) error {
		out, err := c.GetObject(ctx, &s3.GetObjectInput{Bucket: &s.bucket, Key: &key})
		if err != nil {
			return err
		}
		defer out.Body.Close()
		etag = aws.ToString(out.ETag)
		_, err = io.Copy(w, out.Body)
		return err
	})
	return etag, err
}

// explain turns S3 errors into ErrNotFound or a message that says what to fix.
func (s *s3Store) explain(err error) error {
	if err == nil {
		return nil
	}
	var re *awshttp.ResponseError
	if errors.As(err, &re) {
		switch re.HTTPStatusCode() {
		case 404:
			return ErrNotFound
		case 401, 403:
			return fmt.Errorf("access denied to s3://%s (the server's role needs s3:GetObject on the model objects and s3:ListBucket on the prefix): %w", s.bucket, err)
		}
	}
	return err
}
